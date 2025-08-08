"""
fire_poll_worker.py
-------------------

Polls PulsePoint and OpenMHz, transcribes OpenMHz audio using a dual-AI 
process (Whisper + LLM Corrector), correlates events, and pushes unified 
incident reports to a Zapier webhook.

• Env vars required:
    OPENAI_API_KEY        – Your API key from platform.openai.com
    ZAP_CORRELATED_HOOK   – Zapier Webhook for correlated incident events
• Optional env vars:
    OPENMHZ_SYSTEM        – OpenMHz system to watch (default: wakesimul)
    OPENMHZ_TALKGROUPS    – CSV list of talkgroup labels to watch
    PULSEPOINT_URL        – Override PulsePoint URL
"""

from __future__ import annotations
import os, sys, asyncio, datetime, json, re, copy, csv
from typing import Set, Dict, Any, Optional, List, Tuple
from io import BytesIO

# -------- 3rd-party libs --------
from playwright.async_api import async_playwright, TimeoutError as PwTimeout
import aiohttp
import openai
from bson import ObjectId
from rapidfuzz import process, fuzz
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from scipy.io import wavfile
import webrtcvad

# ----------------------------------------------------------------------
# Environment variables
# ----------------------------------------------------------------------
try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    ZAP_CORRELATED_HOOK = os.environ["ZAP_CORRELATED_HOOK"]
except KeyError as missing:
    sys.exit(f"ERROR: you must set {missing} in the environment.")

client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

OPENMHZ_SYSTEM = os.getenv("OPENMHZ_SYSTEM", "wakesimul")
PULSEPOINT_URL = os.getenv(
    "PULSEPOINT_URL", "https://web.pulsepoint.org/?agencies=EMS1209"
)
PULSE_KEYWORDS = {"STRUCTURE FIRE", "FIRE", "FIRE ALARM", "SMOKE", "ALARM"}

# ----------------------------------------------------------------------

def load_talkgroup_list(filepath: str) -> List[str]:
    """Loads the official talkgroup names from the CSV into a list for fuzzy matching."""
    talkgroup_list = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if 'Name' in row and row['Name'].strip():
                    talkgroup_list.append(row['Name'].strip().upper())
        print(f"✅ [TalkgroupLoader] Successfully loaded {len(talkgroup_list)} official talkgroups from {filepath}.")
        return talkgroup_list
    except FileNotFoundError:
        print(f"🚨 [TalkgroupLoader] FATAL: Talkgroup file not found at '{filepath}'. Cannot continue.")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 [TalkgroupLoader] FATAL: Failed to read talkgroup file: {e}")
        sys.exit(1)


async def zapier_post(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]) -> None:
    try:
        async with session.post(url, json=payload) as resp:
            await resp.read()
            if resp.status >= 400:
                body = (await resp.text())[:200]
                print(f"[Zapier] {resp.status} for {url}: {body}")
    except Exception as exc:
        print(f"[Zapier] POST failed: {exc}")

async def correct_transcription_with_llm(raw_text: str, call_type: str, incident_context: str = "") -> str:
    """Uses a powerful LLM to clean up and correct a raw transcription based on the call type and incident context."""
    print(f"[LLM Corrector] Sending '{call_type}' transcription for cleanup...")
    
    prompts = {
        "dispatch": (
            "You are an expert 911 dispatcher's assistant for Wake County, North Carolina. "
            "Your task is to correct and format raw, error-prone radio DISPATCH transcriptions into clean, readable text. "
            "Apply the following rules:\n"
            "- Address Correction: Radio audio is often garbled. Carefully analyze numbers and street names. If an address seems nonsensical (e.g., '10,000 I-11 Roadgate'), reconstruct the most plausible address (e.g., '10511 Rosegate'). Prefer digits for all numbers.\n"
            "- Correct misspellings of local places (e.g., 'Fuquay Varina').\n"
            "- Standardize unit names (e.g., 'engine one' becomes 'Engine 1').\n"
            "- Remove excessive punctuation.\n"
            "- Maintain all critical information."
        ),
        "tactical": (
            "You are an expert fireground radio transcriber. Your task is to correct raw, noisy, and fragmented audio from an active fire scene. "
            "Apply the following rules:\n"
            "- Standardize common phrases: 'nothing showin' or 'nothing shon' becomes 'Nothing Showing'. 'Working fire' is a key phrase. A 'PAR report' means a Personnel Accountability Report.\n"
            "- Correctly format unit names (e.g., 'E1' becomes 'Engine 1').\n"
            "- The text will be fragmented; make it as readable as possible while preserving the core message. Do not add information that is not present.\n"
            f"- **CRITICAL CONTEXT:** The active incident address is: **{incident_context}**. Use this to correct any garbled street names or numbers."
        )
    }
    
    system_prompt = prompts.get(call_type, prompts["dispatch"])
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            temperature=0.0
        )
        corrected_text = response.choices[0].message.content
        print(f"[LLM Corrector] Corrected Text: {corrected_text}")
        return corrected_text.strip()
    except Exception as e:
        print(f"[LLM Corrector] CRITICAL: LLM correction failed: {e}")
        return raw_text

async def transcribe_audio_with_openai(
    audio_url: str, 
    session: aiohttp.ClientSession, 
    call_type: str = "dispatch", 
    incident_context: str = ""
) -> Optional[str]:
    try:
        print(f"[OpenAI] Downloading audio from {audio_url}")
        async with session.get(audio_url) as resp:
            if resp.status != 200:
                print(f"[OpenAI] Error: Failed to download audio. Status: {resp.status}")
                return None
            audio_bytes = await resp.read()

        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes))
        
        MIN_AUDIO_SECONDS = 2.0
        if audio_segment.duration_seconds < MIN_AUDIO_SECONDS:
            print(f"🎤 [Audio] Skipping audio shorter than {MIN_AUDIO_SECONDS} seconds.")
            return None

        if not contains_speech(audio_segment):
            print("🎤 [VAD] No speech detected. Skipping transcription.")
            return None
        
        samples = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate

        print("[Audio] Performing noise reduction...")
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=sample_rate)

        cleaned_audio_bytes = BytesIO()
        wavfile.write(cleaned_audio_bytes, sample_rate, reduced_noise_samples.astype(np.int16))
        cleaned_audio_bytes.seek(0)
        
        print(f"[OpenAI] Cleaned audio created. Sending to Whisper API with '{call_type}' prompt...")
        audio_file = cleaned_audio_bytes
        audio_file.name = "audio.wav"

        prompts = {
            "dispatch": (
                "Residential fire alarm at 123 Main Street, cross of Oak Avenue and Pine Lane. "
                "Engine 1 is on scene. Engine 1, Ladder 2, Battalion 3 responding on channel FIRE OPS 4. "
                "Requesting mutual aid from EMS and PD."
            ),
            "tactical": (
                "Command from Engine 1, we have a working fire. All hands. Engine 2, lay a supply line. "
                "Truck 1, ventilate the roof. All units report a PAR. Three-story residential, nothing showing from the exterior. "
                f"The address is {incident_context}. Checking for extension. Primary search is complete, all clear."
            )
        }
        whisper_prompt = prompts.get(call_type, prompts["dispatch"])
        
        transcription = await client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text",
            prompt=whisper_prompt
        )

        raw_transcription = transcription.strip()
        print(f"[OpenAI] Raw Transcription: {raw_transcription}")
        
        final_transcription = await correct_transcription_with_llm(raw_transcription, call_type, incident_context)
        
        return final_transcription
    except Exception as e:
        print(f"[OpenAI] CRITICAL: Whisper API call failed: {e}")
        return None

def contains_speech(audio_segment: AudioSegment, threshold: float = 0.15) -> bool:
    """Uses VAD to check if an audio segment contains a minimum threshold of speech."""
    SAMPLE_RATE = 32000
    try:
        audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    except Exception as e:
        print(f"⚠️ [VAD] Could not process audio for VAD: {e}")
        return True

    vad = webrtcvad.Vad(3)
    frame_duration_ms = 30
    frame_size = int(SAMPLE_RATE * (frame_duration_ms / 1000.0) * 2)
    
    speech_frames = 0
    total_frames = 0
    
    offset = 0
    while offset + frame_size < len(audio_segment.raw_data):
        frame = audio_segment.raw_data[offset:offset+frame_size]
        total_frames += 1
        if vad.is_speech(frame, SAMPLE_RATE):
            speech_frames += 1
        offset += frame_size

    if total_frames == 0:
        return False
        
    speech_percentage = speech_frames / total_frames
    print(f"🎤 [VAD] Speech percentage: {speech_percentage:.2%}")
    return speech_percentage >= threshold
    
# ----------------------------------------------------------------------
class TalkgroupManager:
    """Manages permanent and temporary talkgroups for monitoring."""
    def __init__(self, initial_talkgroups: Set[str]):
        self.permanent_talkgroups = initial_talkgroups.copy()
        self.temporary_talkgroups: Dict[str, Dict[str, Any]] = {}
        self.TEMP_TALKGROUP_LIFESPAN_MINUTES = 60

    def get_all_monitored_talkgroups(self) -> Set[str]:
        self.cleanup_stale_talkgroups()
        return self.permanent_talkgroups.union(self.temporary_talkgroups.keys())

    def add_temporary_talkgroup(self, talkgroup: str, incident_id: str):
        talkgroup = talkgroup.strip().upper()
        if talkgroup not in self.permanent_talkgroups and talkgroup not in self.temporary_talkgroups:
            print(f"✅ [TalkgroupManager] Adding temporary talkgroup '{talkgroup}' for incident {incident_id}.")
            self.temporary_talkgroups[talkgroup] = {
                "added_at": datetime.datetime.now(datetime.UTC),
                "incident_id": incident_id
            }

    def remove_talkgroups_for_incident(self, incident_id: str):
        """Removes all temporary talkgroups associated with a specific incident ID."""
        tgs_to_remove = [
            tg for tg, data in self.temporary_talkgroups.items()
            if data.get("incident_id") == incident_id
        ]
        if tgs_to_remove:
            print(f"🗑️ [TalkgroupManager] Closing incident {incident_id}. Removing TGs: {tgs_to_remove}")
            for tg in tgs_to_remove:
                del self.temporary_talkgroups[tg]

    def cleanup_stale_talkgroups(self):
        now = datetime.datetime.now(datetime.UTC)
        stale_tgs = [
            tg for tg, data in self.temporary_talkgroups.items()
            if (now - data["added_at"]).total_seconds() > self.TEMP_TALKGROUP_LIFESPAN_MINUTES * 60
        ]
        for tg in stale_tgs:
            print(f"⌛️ [TalkgroupManager] Expiring temporary talkgroup '{tg}'.")
            del self.temporary_talkgroups[tg]
            
# ----------------------------------------------------------------------
class IntelligentIncidentCorrelator:
    def __init__(self, session, hook_url: str, talkgroup_manager: TalkgroupManager, talkgroup_list: List[str]):
        self.session = session
        self.hook_url = hook_url
        self.talkgroup_manager = talkgroup_manager
        self.talkgroup_list = talkgroup_list
        self.pending_incidents: Dict[str, Any] = {}
        self.unmatched_mhz_calls: List[Dict[str, Any]] = []
        
        self.HIGH_CONFIDENCE_THRESHOLD = 85
        self.MEDIUM_CONFIDENCE_THRESHOLD = 70
        self.LOW_CONFIDENCE_THRESHOLD = 55
        self.TIME_CORRELATION_WINDOW_MINUTES = 5
        self.STALE_INCIDENT_MINUTES = 10
        self.STALE_MHZ_MINUTES = 5
        
        self.TRANSCRIPTION_CORRECTIONS = {
            'ZERO': '0', 'OH': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5', 'SIX': '6',
            'SEVEN': '7', 'EIGHT': '8', 'NINE': '9', 'TEN': '10', 'ELEVEN': '11', 'TWELVE': '12', 'FIRST': '1ST',
            'SECOND': '2ND', 'THIRD': '3RD', 'FOURTH': '4TH', 'FIFTH': '5TH', 'SIXTH': '6TH', 'SEVENTH': '7TH',
            'EIGHTH': '8TH', 'NINTH': '9TH', 'TENTH': '10TH'
        }
        self.STREET_ABBREVIATIONS = {
            'STREET': ['ST', 'STR'], 'AVENUE': ['AVE', 'AV'], 'ROAD': ['RD'], 'BOULEVARD': ['BLVD', 'BLV'],
            'DRIVE': ['DR', 'DRV'], 'LANE': ['LN'], 'COURT': ['CT'], 'PLACE': ['PL'], 'CIRCLE': ['CIR'],
            'HIGHWAY': ['HWY'], 'PARKWAY': ['PKWY', 'PKY']
        }
        self.ADDRESS_NORMALIZATIONS = {
            ' U S ': ' US ', ' U.S. ': ' US ', ' HWY ': ' HIGHWAY ', ' ST ': ' STREET ',
            ' AVE ': ' AVENUE ', ' RD ': ' ROAD ', ' BLVD ': ' BOULEVARD ',
            ' DR ': ' DRIVE ', ' LN ': ' LANE ', ' CT ': ' COURT ',
            ' PL ': ' PLACE ', ' CIR ': ' CIRCLE ', ' E ': ' EAST ',
            ' W ': ' WEST ', ' N ': ' NORTH ', ' S ': ' SOUTH '
        }

    def close_incident(self, incident_id: str):
        """Closes out a pending incident, sends a closure alert, and stops TG monitoring."""
        if incident_id in self.pending_incidents:
            print(f"✅ [Correlator] Incident {incident_id} is closed or has disappeared.")
            
            closure_payload = {
                "is_update": True,
                "status": "CLOSED",
                "incident_id": incident_id,
                "transcription": "This incident has been closed or is no longer active."
            }
            print(f"🚀 [Correlator] Sending CLOSED update for {incident_id}.")
            asyncio.create_task(zapier_post(self.session, self.hook_url, closure_payload))
            
            self.talkgroup_manager.remove_talkgroups_for_incident(incident_id)
            del self.pending_incidents[incident_id]

    def _normalize_text(self, text: str) -> str:
        if not text: return ""
        text = text.upper().strip()
        
        padded_text = f" {text} "
        for error, correction in self.ADDRESS_NORMALIZATIONS.items():
            padded_text = padded_text.replace(error, correction)
        text = padded_text.strip()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        words = text.split()
        corrected_words = [self.TRANSCRIPTION_CORRECTIONS.get(word, word) for word in words]
        return ' '.join(corrected_words)

    def _extract_address_components(self, address: str) -> Dict[str, Any]:
        normalized = self._normalize_text(address)
        numbers = re.findall(r'\b\d+\b', normalized)
        words = normalized.split()
        street_words = [word for word in words if not word.isdigit() and not any(word in group for group in self.STREET_ABBREVIATIONS.values())]
        is_intersection = bool(re.search(r'\b(AND|&|\bAT\b|\/)\b', normalized))
        return {'full': normalized, 'numbers': numbers, 'street_words': street_words, 'is_intersection': is_intersection, 'raw': address}

    def _extract_transcript_addresses(self, transcript: str) -> List[Dict[str, Any]]:
        normalized = self._normalize_text(transcript)
        addresses = []
        for pat in [r'\b(\d+)\s+([A-Z]+(?:\s+[A-Z]+)*)', r'([A-Z]+(?:\s+[A-Z]+)*)\s+(?:AND|&|AT)\s+([A-Z]+(?:\s+[A-Z]+)*)']:
            for match in re.finditer(pat, normalized):
                addresses.append(self._extract_address_components(match.group(0)))
        return addresses

    def _calculate_component_similarity(self, addr1: Dict[str, Any], addr2: Dict[str, Any]) -> float:
        score, weight = 0.0, 0.0
        
        if addr1['numbers'] and addr2['numbers']:
            num_str1 = "".join(addr1['numbers'])
            num_str2 = "".join(addr2['numbers'])
            number_score = fuzz.ratio(num_str1, num_str2) / 100.0
            score += number_score * 50
            weight += 50
            
        if addr1['street_words'] and addr2['street_words']:
            street_scores = []
            for word1 in addr1['street_words']:
                best_match = process.extractOne(word1, addr2['street_words'], scorer=fuzz.partial_ratio)
                if best_match:
                    street_scores.append(best_match[1])
            if street_scores:
                score += (sum(street_scores) / len(street_scores) / 100) * 30
                weight += 30

        score += (fuzz.token_set_ratio(addr1['full'], addr2['full']) / 100) * 20
        weight += 20
        
        return (score / weight * 100) if weight > 0 else 0

    def _find_best_match_advanced(self, address: str, transcript: str) -> Tuple[Optional[float], str]:
        incident_comps = self._extract_address_components(address)
        transcript_addrs = self._extract_transcript_addresses(transcript)
        if not transcript_addrs:
            score = fuzz.token_set_ratio(self._normalize_text(address), self._normalize_text(transcript))
            return (score, "simple_match") if score >= self.LOW_CONFIDENCE_THRESHOLD else (None, "")
        
        best_score = max((self._calculate_component_similarity(incident_comps, ta) for ta in transcript_addrs), default=0)
        return (best_score, "component_match") if best_score >= self.LOW_CONFIDENCE_THRESHOLD else (None, "")

    def _is_time_correlated(self, time1: datetime.datetime, time2: datetime.datetime) -> bool:
        return abs((time1 - time2).total_seconds()) <= self.TIME_CORRELATION_WINDOW_MINUTES * 60

    def _calculate_confidence_score(self, score: float, time_corr: bool, has_kw: bool) -> str:
        if score >= self.HIGH_CONFIDENCE_THRESHOLD and time_corr: return "HIGH"
        if score >= self.MEDIUM_CONFIDENCE_THRESHOLD and (time_corr or has_kw): return "MEDIUM"
        return "LOW"

    def _check_keywords(self, transcript: str, incident_type: str) -> bool:
        t_lower, i_lower = transcript.lower(), incident_type.lower()
        kw_map = {'structure fire': ['structure', 'fire', 'flames', 'smoke'], 'fire alarm': ['alarm', 'bells'], 'smoke': ['smoke', 'odor'], 'fire': ['fire']}
        for key, kws in kw_map.items():
            if key in i_lower: return any(kw in t_lower for kw in kws)
        return False

    def _extract_and_process_hot_talkgroup(self, transcript: str, incident: Dict[str, Any]):
        clean_transcript = ' '.join(transcript.split())
        
        # General regex to capture potential misspellings (e.g., PAC for TAC)
        match = re.search(r'(?:on channel|on|channel|switch to)\s+([a-zA-Z\s-]+\d+)', clean_transcript, re.IGNORECASE)
        
        # Fallback for cases where only a number is spoken
        if not match:
            match = re.search(r'(?:on channel|on|channel|switch to)\s+(\d+)', clean_transcript, re.IGNORECASE)

        if not match:
            print(f"DEBUG: Talkgroup regex failed to match on cleaned transcript: '{clean_transcript}'")
            return

        extracted_part = re.sub(r'\s+', ' ', match.group(1).strip().upper())
        official_tg = None

        if extracted_part.isdigit():
            constructed_tg = f"FIRE OPS {extracted_part}"
            if constructed_tg in self.talkgroup_list:
                official_tg = constructed_tg
                print(f"✅ [Correlator] Interpreted channel '{extracted_part}' as default '{official_tg}'")
        else:
            best_match = process.extractOne(
                extracted_part,
                self.talkgroup_list,
                scorer=fuzz.token_set_ratio,
                score_cutoff=80 # More lenient for misspellings like PAC vs TAC
            )
            if best_match:
                matched_tg, score, _ = best_match
                normalized_tg = matched_tg
                number_match = re.search(r'\d+', matched_tg)
                if number_match:
                    number = number_match.group(0)
                    if "OPS" in matched_tg:
                        normalized_tg = f"FIRE OPS {number}"
                    elif "TAC" in matched_tg:
                        normalized_tg = f"TAC {number}"
                
                print(f"✅ [Correlator] Matched '{extracted_part}' to '{matched_tg}', normalized to '{normalized_tg}' (Score: {score:.0f})")
                official_tg = normalized_tg

        if official_tg:
            incident['hot_talkgroups'].add(official_tg)
            self.talkgroup_manager.add_temporary_talkgroup(official_tg, incident['pulsepoint_data']['incident_id'])
        else:
            print(f"❌ [Correlator] Extracted TG '{extracted_part}' could not be matched/validated.")
            print(f"   [Debug] Official List Searched: {self.talkgroup_list}")

    def process_mhz_call(self, mhz_payload: Dict[str, Any]):
        talkgroup = mhz_payload.get("talkgroup", "").upper()
        transcript = mhz_payload.get("transcription", "")

        if talkgroup in self.talkgroup_manager.temporary_talkgroups:
            incident_id = self.talkgroup_manager.temporary_talkgroups[talkgroup]['incident_id']
            if incident_id in self.pending_incidents:
                incident = self.pending_incidents[incident_id]
                incident['openmhz_calls'].append({'match_confidence': 'TACTICAL', **mhz_payload})

                update_payload = {
                    "is_update": True,
                    "incident_id": incident_id,
                    "talkgroup": mhz_payload.get("talkgroup"),
                    "audio_link": mhz_payload.get("audio_link"),
                    "transcription": mhz_payload.get("transcription")
                }
                print(f"🚀 [Correlator] Sending transcription-only update for {incident_id}.")
                asyncio.create_task(zapier_post(self.session, self.hook_url, update_payload))
                return

        if not transcript: return
        
        call_time = datetime.datetime.fromisoformat(mhz_payload.get('seen_at', datetime.datetime.now(datetime.UTC).isoformat()))
        best_match, best_score, best_confidence = None, 0, "NONE"
        
        for incident_id, incident in self.pending_incidents.items():
            if incident.get("processed"): continue
            
            match_score, _ = self._find_best_match_advanced(incident['pulsepoint_data']['address'], transcript)
            if not match_score: continue
            
            time_corr = self._is_time_correlated(incident['created_at'], call_time)
            has_kw = self._check_keywords(transcript, incident['pulsepoint_data']['type'])
            confidence = self._calculate_confidence_score(match_score, time_corr, has_kw)
            
            adj_score = match_score + (10 if time_corr else 0) + (5 if has_kw else 0)
            if adj_score > best_score:
                best_score, best_match, best_confidence = adj_score, incident, confidence
        
        print(f"[Correlator] Address match score: {best_score:.1f}, Confidence: {best_confidence}")
        
        if best_match and best_confidence in ["HIGH", "MEDIUM"]:
            print(f"[Correlator] MATCH FOUND! Confidence: {best_confidence}")
            best_match['openmhz_calls'].append({'match_confidence': best_confidence, 'match_score': best_score, **mhz_payload})
            self._extract_and_process_hot_talkgroup(transcript, best_match)
            if not best_match.get("processed"):
                best_match["processed"] = True
                self._send_correlated_alert(best_match)
        else:
            print("[Correlator] No confident match found. Storing as unmatched.")
            mhz_payload['received_at'] = call_time
            self.unmatched_mhz_calls.append(mhz_payload)
    
    def add_pulsepoint_incident(self, payload: Dict[str, Any]):
        incident_id = payload['incident_id']
        if incident_id in self.pending_incidents:
            if not self.pending_incidents[incident_id].get("processed"):
                print(f"[Correlator] UPDATING incident {incident_id} type to '{payload['type']}'.")
                self.pending_incidents[incident_id]['pulsepoint_data'] = payload
            return
            
        print(f"[Correlator] New pending incident: {incident_id}")
        new_incident = {"pulsepoint_data": payload, "openmhz_calls": [], "hot_talkgroups": set(), "processed": False, "created_at": datetime.datetime.now(datetime.UTC)}
        self.pending_incidents[incident_id] = new_incident
        
        newly_matched, still_unmatched = [], []
        for call in self.unmatched_mhz_calls:
            transcript = call.get("transcription", "")
            if not transcript:
                still_unmatched.append(call)
                continue
            
            match_score, _ = self._find_best_match_advanced(payload['address'], transcript)
            if not match_score:
                still_unmatched.append(call)
                continue
            
            time_corr = self._is_time_correlated(new_incident['created_at'], call['received_at'])
            has_kw = self._check_keywords(transcript, payload['type'])
            confidence = self._calculate_confidence_score(match_score, time_corr, has_kw)
            
            if confidence in ["HIGH", "MEDIUM"]:
                adj_score = match_score + (10 if time_corr else 0) + (5 if has_kw else 0)
                print(f"[Correlator] RETROACTIVE MATCH! Score {adj_score:.1f}, Confidence {confidence}")
                call.update({'match_confidence': confidence, 'match_score': adj_score})
                newly_matched.append(call)
                self._extract_and_process_hot_talkgroup(transcript, new_incident)
            else:
                still_unmatched.append(call)

        self.unmatched_mhz_calls = still_unmatched
        if newly_matched:
            new_incident['openmhz_calls'].extend(newly_matched)
            if not new_incident.get("processed"):
                new_incident["processed"] = True
                self._send_correlated_alert(new_incident)

    def _prepare_payload_for_json(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(incident_data)
        if isinstance(payload.get('hot_talkgroups'), set):
            payload['hot_talkgroups'] = sorted(list(payload['hot_talkgroups']))
        for key in ['created_at', 'received_at', 'seen_at']:
            if isinstance(payload.get(key), datetime.datetime):
                payload[key] = payload[key].isoformat()
        for call in payload.get('openmhz_calls', []):
            for key in ['received_at', 'seen_at']:
                if isinstance(call.get(key), datetime.datetime): call[key] = call[key].isoformat()
        return payload

    def _send_correlated_alert(self, incident: Dict[str, Any]):
        """Sends the full incident payload to the configured webhook."""
        print(f"🚀 [Correlator] Preparing and sending FULL alert for {incident['pulsepoint_data']['incident_id']}.")
        json_safe_payload = self._prepare_payload_for_json(incident)
        json_safe_payload["is_update"] = False
        asyncio.create_task(zapier_post(self.session, self.hook_url, json_safe_payload))
        
    async def cleanup_stale_items(self):
        while True:
            await asyncio.sleep(60)
            now = datetime.datetime.now(datetime.UTC)
            stale_ids = [k for k, v in self.pending_incidents.items() if not v.get('openmhz_calls') and (now - v["created_at"]).total_seconds() > self.STALE_INCIDENT_MINUTES * 60]
            if stale_ids:
                print(f"[Correlator] Cleaning up {len(stale_ids)} stale incidents.")
                for iid in stale_ids: del self.pending_incidents[iid]
            
            original_count = len(self.unmatched_mhz_calls)
            self.unmatched_mhz_calls = [c for c in self.unmatched_mhz_calls if "received_at" in c and (now - c["received_at"]).total_seconds() <= self.STALE_MHZ_MINUTES * 60]
            if (cleaned_count := original_count - len(self.unmatched_mhz_calls)) > 0:
                print(f"[Correlator] Cleaned up {cleaned_count} stale unmatched audio calls.")

# ----------------------------------------------------------------------
class PulsePointMonitor:
    CARD_SEL = 'tr:has(img[src*="/images/respond_icons/"])'
    def __init__(self, page, correlator: IntelligentIncidentCorrelator):
        self.page, self.correlator, self.seen_ids = page, correlator, set()

    def _parse_card_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parses card text, identifying if it's a new incident or a closed one."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 3: return None
        
        incident_type, time_str, address = lines[0], lines[1], lines[2]

        if any("CLOSED" in line.upper() for line in lines):
            return {"status": "CLOSED", "address": address}

        if not re.match(r'^\d{1,2}:\d{2}\s(A|P)M$', time_str.upper()): return None
        if "DURATION" in address.upper(): return None
        
        return {"status": "NEW", "type": incident_type, "time": time_str, "address": address}

    async def run(self) -> None:
        print("PulsePoint monitor started")
        await self.page.goto(PULSEPOINT_URL, timeout=60_000, wait_until="domcontentloaded")
        try:
            await self.page.locator(self.CARD_SEL).first.wait_for(timeout=60_000)
            print("[PulsePoint] Priming initial incidents...")
            for card in await self.page.locator(self.CARD_SEL).all():
                try:
                    full_text = await card.inner_text(timeout=2000)
                    if parsed := self._parse_card_text(full_text):
                        if parsed['status'] == 'NEW':
                           self.seen_ids.add(self.correlator._normalize_text(parsed['address']))
                except PwTimeout:
                    print("[PulsePoint] A card disappeared during priming. Skipping.")
            print(f"[PulsePoint] Primed with {len(self.seen_ids)} incidents. Monitoring.")
        except Exception as e:
            print(f"[PulsePoint] CRITICAL: Could not prime incidents: {e}")
            await self.page.screenshot(path="pulsepoint_error.png")
            return
        while True:
            try:
                await asyncio.sleep(15)
                
                current_incident_ids = set()
                cards = await self.page.locator(self.CARD_SEL).all()
                
                for card in cards:
                    try:
                        full_text = await card.inner_text(timeout=5000)
                        parsed_data = self._parse_card_text(full_text)
                        if not parsed_data: continue
                        
                        incident_id = self.correlator._normalize_text(parsed_data['address'])
                        current_incident_ids.add(incident_id)

                        if parsed_data['status'] == 'CLOSED':
                            self.correlator.close_incident(incident_id)
                            if incident_id in self.seen_ids:
                                self.seen_ids.remove(incident_id)
                            continue

                        if incident_id in self.seen_ids: continue
                        if 'type' in parsed_data and not any(key in parsed_data['type'].upper() for key in PULSE_KEYWORDS): continue
                        
                        self.seen_ids.add(incident_id)
                        payload = {"incident_id": incident_id, "type": parsed_data['type'], "time_str": parsed_data['time'], "address": parsed_data['address'], "scraped_at": datetime.datetime.now(datetime.UTC).isoformat()}
                        self.correlator.add_pulsepoint_incident(payload)
                    except PwTimeout: continue
                
                pending_ids = set(self.correlator.pending_incidents.keys())
                disappeared_ids = pending_ids - current_incident_ids
                if disappeared_ids:
                    print(f"[PulsePoint] Detected {len(disappeared_ids)} disappeared incidents.")
                    for incident_id in disappeared_ids:
                        self.correlator.close_incident(incident_id)
                        if incident_id in self.seen_ids:
                            self.seen_ids.remove(incident_id)

            except Exception as exc:
                print(f"[PulsePoint] Error in monitor loop: {exc}.")
                await asyncio.sleep(5)

# ----------------------------------------------------------------------
class OpenMHzMonitor:
    BASE_URL = f"https://openmhz.com/system/{OPENMHZ_SYSTEM}"
    def __init__(self, page, session: aiohttp.ClientSession, correlator: IntelligentIncidentCorrelator, talkgroup_manager: TalkgroupManager):
        self.page, self.session, self.correlator, self.seen_ids = page, session, correlator, set()
        self.talkgroup_manager = talkgroup_manager
        
    async def _initial_setup(self) -> None:
        print("[OpenMHz] Performing initial setup...")
        await self.page.goto(self.BASE_URL, wait_until="domcontentloaded", timeout=60_000)
        await self.page.wait_for_selector("tbody tr[data-callid]", timeout=30_000)
        print("[OpenMHz] Priming initial calls...")
        for row in await self.page.locator("tbody tr[data-callid]").all():
            if call_id := await row.get_attribute("data-callid"): self.seen_ids.add(call_id)
        print(f"[OpenMHz] Primed with {len(self.seen_ids)} calls. Monitoring for new events.")
        
    async def run(self) -> None:
        print("OpenMHz monitor started")
        try: await self._initial_setup()
        except Exception as exc:
            print(f"[OpenMHz] CRITICAL: Initial setup failed: {exc}")
            return
        while True:
            try:
                await asyncio.sleep(20)
                monitored_tgs = self.talkgroup_manager.get_all_monitored_talkgroups()
                print(f"[OpenMHz] Reloading. Monitoring {len(monitored_tgs)} TGs: {sorted(list(monitored_tgs))}")
                await self.page.reload(wait_until="domcontentloaded")
                await self.page.wait_for_selector("tbody tr[data-callid]", timeout=30_000)
                
                modal_locator = self.page.locator('.ui.page.modals.dimmer.visible.active')
                if await modal_locator.is_visible(timeout=1000):
                    print("[OpenMHz] Blocking modal detected. Pressing Escape to dismiss.")
                    await self.page.keyboard.press("Escape")
                    await modal_locator.wait_for(state="hidden", timeout=5000)
                    print("[OpenMHz] Modal dismissed.")
                
                for row in reversed(await self.page.locator("tbody tr[data-callid]").all()):
                    try:
                        call_id = await row.get_attribute("data-callid")
                        if not call_id or call_id in self.seen_ids: continue
                        
                        tg = (await row.locator("td:nth-child(3)").inner_text(timeout=1000)).strip().upper()
                        if tg not in monitored_tgs: continue
                        
                        self.seen_ids.add(call_id)
                        print(f"[OpenMHz] Found new call {call_id} on '{tg}'. Processing...")
                        
                        call_type = "tactical" if tg in self.talkgroup_manager.temporary_talkgroups else "dispatch"
                        incident_id = None
                        if call_type == "tactical":
                            incident_id = self.talkgroup_manager.temporary_talkgroups[tg]['incident_id']

                        await row.click(timeout=5000)
                        
                        dlink = self.page.locator('a.item[href$=".m4a"]')
                        await dlink.wait_for(state="visible", timeout=10_000)
                        m4a_url = await dlink.get_attribute("href")
                        
                        transcription = None
                        if m4a_url:
                            incident_context = ""
                            if incident_id and incident_id in self.correlator.pending_incidents:
                                incident_context = self.correlator.pending_incidents[incident_id]['pulsepoint_data']['address']
                            
                            transcription = await transcribe_audio_with_openai(m4a_url, self.session, call_type, incident_context)
                        
                        try: await self.page.keyboard.press("Escape")
                        except Exception: pass
                        
                        if not transcription: continue
                        
                        payload = {"call_id": call_id, "talkgroup": tg, "audio_link": m4a_url, "transcription": transcription, "seen_at": datetime.datetime.now(datetime.UTC).isoformat()}
                        self.correlator.process_mhz_call(payload)
                        print(f"[OpenMHz] Successfully processed call {call_id}. Breaking loop for this cycle.")
                        break
                    except Exception as e:
                        print(f"[OpenMHz] Error processing call row: {e}. Continuing.")
            except PwTimeout as exc:
                print(f"🚨 [OpenMHz] CRITICAL: Page failed to load the call list in time. Taking a screenshot.")
                await self.page.screenshot(path="openmhz_error.png")
                print(f"[OpenMHz] {exc}")
                await asyncio.sleep(30)
            except Exception as exc:
                print(f"[OpenMHz] Error in main loop: {exc}")
                await asyncio.sleep(10)

# ----------------------------------------------------------------------
async def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    
    talkgroup_list = load_talkgroup_list("Tacops.xlsx - Sheet1.csv")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), headers=headers) as session:
        
        initial_tgs = {tg.strip().upper() for tg in os.getenv("OPENMHZ_TALKGROUPS", "RFD ALERT,WC FD DISP,WC FD ALERT").split(",") if tg.strip()}
        talkgroup_manager = TalkgroupManager(initial_tgs)
        correlator = IntelligentIncidentCorrelator(session, ZAP_CORRELATED_HOOK, talkgroup_manager, talkgroup_list)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False, channel="chrome")
            pulse_page = await browser.new_page()
            mhz_page = await browser.new_page()

            tasks = [
                asyncio.create_task(correlator.cleanup_stale_items()),
                asyncio.create_task(PulsePointMonitor(pulse_page, correlator).run()),
                asyncio.create_task(OpenMHzMonitor(mhz_page, session, correlator, talkgroup_manager).run()),
            ]
            await asyncio.gather(*tasks)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down …")
