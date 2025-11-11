#!/usr/bin/env python3
"""
Streamlit app for iCalendar to CSV Extractor with Patient Matching
"""

import streamlit as st
import csv
import re
from datetime import datetime, timedelta
from icalendar import Calendar
import pytz
from typing import List, Dict, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher
import logging
from pathlib import Path
import unicodedata
import io
import zipfile
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="iCalendar Appointment Processor",
    page_icon="📅",
    layout="wide"
)

# Setup logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for storing results
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'files_to_download' not in st.session_state:
    st.session_state.files_to_download = {}

# Helper functions from original script
_WS_RE = re.compile(r"\s+")

def _normalize_token(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("\u200B", "")   # zero-width space
    s = s.replace("\u200C", "")   # ZWNJ
    s = s.replace("\u200D", "")   # ZWJ
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s

_CLEAN_NOISE_RE = re.compile(r"(?:\bTMS\b|#\d+|\d+/\d+|\bF\d{2}\.\d\b|\b[FR]\d{2}\.\d\b|[()])", re.IGNORECASE)

def _clean_person_token(s: str) -> str:
    s = _CLEAN_NOISE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _surname_key(s: str) -> str:
    n = _normalize_token(s)

    # Armenian harmonization:
    if n.endswith("ielyan"):
        n = n[:-6] + "ilyan"
    elif n.endswith("yelyan"):
        n = n[:-6] + "lyan"
    elif n.endswith("elyan"):
        n = n[:-5] + "lyan"

    # General Slavic/Armenian feminine/masculine harmonization
    repl = [
        ("skaya", "sky"), ("tskaya", "tsky"), ("vskaya", "vsky"), ("zkaya", "zky"),
        ("ckaya", "cky"), ("shaya", "shay"), ("chaya", "chay"), ("zhaya", "zhay"),
        ("aya", "y"), ("ova", "ov"), ("eva", "ev"), ("ina", "in"), ("kina", "kin"), ("yina", "yin"),
    ]
    for suf, to in repl:
        if n.endswith(suf):
            return n[: -len(suf)] + to
    return n

class PatientData:
    """Patient information"""
    def __init__(self, first_name: str, last_name: str, prn: str, insurance: str, doctor: str):
        self.first_name = first_name.strip()
        self.last_name = last_name.strip()
        self.prn = prn.strip()
        self.insurance = insurance.strip()
        self.doctor = doctor.strip()
    
    def full_name(self) -> str:
        """Return name in 'Last Name, First Name' format"""
        return f"{self.last_name}, {self.first_name}"

def parse_ical_file(ical_content: bytes) -> Calendar:
    """Parse the iCalendar file and return Calendar object"""
    try:
        calendar = Calendar.from_ical(ical_content)
        return calendar
    except Exception as e:
        raise Exception(f"Error parsing iCalendar file: {str(e)}")

def is_monday_or_friday(dt: datetime) -> bool:
    """Check if the datetime falls on Monday (0) or Friday (4)"""
    return dt.weekday() in [0, 4]

def is_in_target_months(dt: datetime, target_months: List[int], target_year: int) -> bool:
    """Check if the datetime is in the target months and year"""
    return dt.year == target_year and dt.month in target_months

def extract_appointments(calendar: Calendar, months: List[int], year: int) -> List[Dict]:
    """Extract appointments that fall on Monday/Friday in specified months"""
    appointments = []
    
    for component in calendar.walk():
        if component.name == "VEVENT":
            summary = str(component.get('summary', 'No Title'))
            description = str(component.get('description', ''))
            location = str(component.get('location', ''))
            
            dtstart = component.get('dtstart')
            if dtstart is None:
                continue
                
            start_dt = dtstart.dt
            
            if isinstance(start_dt, datetime):
                event_datetime = start_dt
            else:
                event_datetime = datetime.combine(start_dt, datetime.min.time())
            
            if event_datetime.tzinfo is None:
                event_datetime = pytz.UTC.localize(event_datetime)
            
            dtend = component.get('dtend')
            if dtend:
                end_dt = dtend.dt
                if isinstance(end_dt, datetime):
                    end_datetime = end_dt
                else:
                    end_datetime = datetime.combine(end_dt, datetime.min.time())
                
                if end_datetime.tzinfo is None:
                    end_datetime = pytz.UTC.localize(end_datetime)
            else:
                end_datetime = event_datetime + timedelta(hours=1)
            
            rrule = component.get('rrule')
            if rrule:
                from dateutil.rrule import rrulestr
                
                rrule_str = f"DTSTART:{event_datetime.strftime('%Y%m%dT%H%M%SZ')}\n"
                rrule_str += f"RRULE:{rrule.to_ical().decode('utf-8')}"
                
                try:
                    rule = rrulestr(rrule_str)
                    start_of_year = datetime(year, 1, 1, tzinfo=pytz.UTC)
                    end_of_year = datetime(year + 1, 1, 1, tzinfo=pytz.UTC)
                    
                    for occurrence in rule.between(start_of_year, end_of_year, inc=True):
                        if is_monday_or_friday(occurrence) and is_in_target_months(occurrence, months, year):
                            duration = end_datetime - event_datetime
                            occurrence_end = occurrence + duration
                            
                            appointments.append({
                                'title': summary,
                                'start_date': occurrence.strftime('%m/%d/%Y'),
                                'start_time': occurrence.strftime('%H:%M'),
                                'end_time': occurrence_end.strftime('%H:%M'),
                                'day_of_week': occurrence.strftime('%A'),
                                'location': location,
                                'description': description.replace('\n', ' ').replace('\r', '')
                            })
                except Exception as e:
                    logger.warning(f"Could not process recurring event '{summary}': {str(e)}")
            else:
                if is_monday_or_friday(event_datetime) and is_in_target_months(event_datetime, months, year):
                    appointments.append({
                        'title': summary,
                        'start_date': event_datetime.strftime('%m/%d/%Y'),
                        'start_time': event_datetime.strftime('%H:%M'),
                        'end_time': end_datetime.strftime('%H:%M'),
                        'day_of_week': event_datetime.strftime('%A'),
                        'location': location,
                        'description': description.replace('\n', ' ').replace('\r', '')
                    })
    
    appointments.sort(key=lambda x: (datetime.strptime(x['start_date'], '%m/%d/%Y'), x['start_time']))
    return appointments

def load_patient_list(excel_content: bytes) -> Dict[str, PatientData]:
    """Load patient list from Excel file content"""
    try:
        df = pd.read_excel(io.BytesIO(excel_content), header=None)
        patients: Dict[str, PatientData] = {}

        for _, row in df.iterrows():
            try:
                raw_last = str(row[0])
                raw_first = str(row[1]) if len(row) > 1 else ""
                raw_prn = str(row[2]) if len(row) > 2 else ""
                raw_insurance = str(row[3]) if len(row) > 3 else ""
                raw_doctor = str(row[4]) if len(row) > 4 else ""

                if pd.isna(row[0]):
                    continue

                first_name = raw_first.strip() if not pd.isna(row[1]) else ""
                last_name = raw_last.strip()
                prn = raw_prn.strip() if not pd.isna(row[2]) else ""
                insurance = raw_insurance.strip() if not pd.isna(row[3]) else ""
                doctor = raw_doctor.strip() if not pd.isna(row[4]) else ""

                patient = PatientData(first_name, last_name, prn, insurance, doctor)
                full_name = patient.full_name()
                patients[full_name] = patient
                
            except Exception as e:
                logger.warning(f"Could not process row: {e}")
                continue

        return patients
    except Exception as e:
        raise Exception(f"Error loading patient list: {str(e)}")

def extract_name_and_codes(appointment_title: str) -> List[Tuple[str, str]]:
    """Extract patient names and procedure codes from appointment title"""
    title = appointment_title.strip()
    
    if not title or title == "No Title":
        return []
    
    # Extract metadata (TMS/CPT codes)
    metadata_parts = []
    
    # Pattern 1: TMS#number
    tms_matches = re.findall(r'\bTMS#(\d+)\b', title, re.IGNORECASE)
    for tms in tms_matches:
        metadata_parts.append(f"TMS#{tms}")
    
    # Pattern 2: CPT code format (e.g., 99213, 90837)
    cpt_matches = re.findall(r'\b(9\d{4})\b', title)
    for cpt in cpt_matches:
        if cpt not in tms_matches:  # Avoid duplicates
            metadata_parts.append(f"CPT:{cpt}")
    
    # Pattern 3: F codes (e.g., F43.10)
    f_code_matches = re.findall(r'\b(F\d{2}\.\d{1,2})\b', title, re.IGNORECASE)
    for f_code in f_code_matches:
        metadata_parts.append(f_code.upper())
    
    metadata = ", ".join(metadata_parts) if metadata_parts else ""
    
    # Clean title for name extraction
    clean_title = title
    for pattern in [r'\bTMS#\d+\b', r'\b9\d{4}\b', r'\bF\d{2}\.\d{1,2}\b',
                   r'\(.*?\)', r'\[.*?\]', r'\d{1,2}/\d{1,2}', r'#\d+']:
        clean_title = re.sub(pattern, ' ', clean_title, flags=re.IGNORECASE)
    
    # Handle "and" or "&" for multiple patients
    separators = [' and ', ' & ', ', ']
    names = [clean_title]
    
    for sep in separators:
        new_names = []
        for name in names:
            if sep in name.lower() or sep.strip() in name:
                parts = re.split(re.escape(sep), name, flags=re.IGNORECASE)
                new_names.extend(parts)
            else:
                new_names.append(name)
        names = new_names
    
    results = []
    for name in names:
        name = re.sub(r'\s+', ' ', name).strip()
        if name and not name.isdigit():
            results.append((name, metadata))
    
    return results if results else [(clean_title.strip(), metadata)]

def find_best_patient_match(name: str, patients: Dict[str, PatientData], 
                           threshold: float = 0.75) -> Optional[Tuple[PatientData, float, bool]]:
    """Find the best matching patient for a given name using fuzzy matching"""
    
    if not name or not patients:
        return None
    
    name = _clean_person_token(name)
    name_norm = _normalize_token(name)
    
    # Try exact match first
    for patient_key, patient in patients.items():
        if name_norm == _normalize_token(patient.full_name()):
            return (patient, 1.0, False)
    
    # Parse the input name
    name_parts = name.split(',')
    if len(name_parts) == 2:
        input_last = name_parts[0].strip()
        input_first = name_parts[1].strip()
    else:
        name_tokens = name.split()
        if len(name_tokens) >= 2:
            input_last = name_tokens[-1]
            input_first = ' '.join(name_tokens[:-1])
        elif len(name_tokens) == 1:
            input_last = name_tokens[0]
            input_first = ""
        else:
            return None
    
    input_last_norm = _normalize_token(input_last)
    input_first_norm = _normalize_token(input_first)
    input_surname_key = _surname_key(input_last)
    
    best_match = None
    best_score = 0.0
    candidates = []
    
    for patient_key, patient in patients.items():
        patient_last_norm = _normalize_token(patient.last_name)
        patient_first_norm = _normalize_token(patient.first_name)
        patient_surname_key = _surname_key(patient.last_name)
        
        # Calculate last name similarity
        if input_surname_key == patient_surname_key:
            last_score = 1.0
        else:
            last_score = SequenceMatcher(None, input_last_norm, patient_last_norm).ratio()
        
        # Calculate first name similarity
        if not input_first_norm or not patient_first_norm:
            first_score = 0.5 if not input_first_norm else 0.0
        elif input_first_norm == patient_first_norm:
            first_score = 1.0
        elif input_first_norm[0] == patient_first_norm[0]:
            first_score = 0.7
        else:
            first_score = SequenceMatcher(None, input_first_norm, patient_first_norm).ratio()
        
        # Combined score (weighted: last name more important)
        combined_score = (last_score * 0.7) + (first_score * 0.3)
        
        if combined_score >= threshold:
            candidates.append((patient, combined_score))
            if combined_score > best_score:
                best_score = combined_score
                best_match = patient
    
    if best_match:
        # Check for ambiguous matches
        is_ambiguous = len([c for c in candidates if abs(c[1] - best_score) < 0.05]) > 1
        return (best_match, best_score, is_ambiguous)
    
    return None

def process_appointments_with_patients(appointments: List[Dict], 
                                      patients: Dict[str, PatientData]) -> Tuple[List[Dict], List[Dict], Dict]:
    """Process appointments and match with patient records"""
    
    matched = []
    unmatched = []
    stats = {
        'total_appointments': len(appointments),
        'total_names_extracted': 0,
        'matched_patients': 0,
        'unmatched_entries': 0,
        'split_entries': 0,
        'metadata_extracted': 0,
        'fuzzy_matches': 0,
        'ambiguous_matches': 0,
        'rejected_low_confidence': 0
    }
    
    for apt in appointments:
        original_title = apt['title'].strip()
        
        if not original_title or original_title == "No Title":
            continue
        
        names_and_codes = extract_name_and_codes(original_title)
        stats['total_names_extracted'] += len(names_and_codes)
        
        if len(names_and_codes) > 1:
            stats['split_entries'] += 1
        
        for name, metadata in names_and_codes:
            if metadata:
                stats['metadata_extracted'] += 1
            
            match_result = find_best_patient_match(name, patients)
            
            if match_result:
                patient, confidence, is_ambiguous = match_result
                
                if confidence < 0.75:
                    stats['rejected_low_confidence'] += 1
                    unmatched_entry = {
                        'original_name': name,
                        'date': apt['start_date'],
                        'start_time': apt['start_time'],
                        'end_time': apt['end_time'],
                        'day_of_week': apt['day_of_week'],
                        'full_title': original_title,
                        'codes': metadata
                    }
                    unmatched.append(unmatched_entry)
                    stats['unmatched_entries'] += 1
                    continue
                
                matched_entry = {
                    'name': patient.full_name(),
                    'date': apt['start_date'],
                    'start_time': apt['start_time'],
                    'end_time': apt['end_time'],
                    'day_of_week': apt['day_of_week'],
                    'prn': patient.prn,
                    'insurance': patient.insurance,
                    'doctor': patient.doctor,
                    'codes': metadata,
                    'original_title': original_title,
                    'confidence': f"{confidence*100:.1f}%"
                }
                matched.append(matched_entry)
                stats['matched_patients'] += 1
                
                if confidence < 1.0:
                    stats['fuzzy_matches'] += 1
                if is_ambiguous:
                    stats['ambiguous_matches'] += 1
            else:
                unmatched_entry = {
                    'original_name': name,
                    'date': apt['start_date'],
                    'start_time': apt['start_time'],
                    'end_time': apt['end_time'],
                    'day_of_week': apt['day_of_week'],
                    'full_title': original_title,
                    'codes': metadata
                }
                unmatched.append(unmatched_entry)
                stats['unmatched_entries'] += 1
    
    return matched, unmatched, stats

def create_csv_content(matched: List[Dict]) -> str:
    """Create CSV content from matched appointments"""
    output = io.StringIO()
    
    if not matched:
        return ""
    
    fieldnames = ['name', 'date', 'start_time', 'end_time', 'day_of_week', 
                  'prn', 'insurance', 'doctor', 'codes', 'original_title', 'confidence']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(matched)
    
    return output.getvalue()

def create_excel_content(unmatched: List[Dict]) -> bytes:
    """Create Excel content from unmatched entries"""
    if not unmatched:
        return b""
    
    df = pd.DataFrame(unmatched)
    output = io.BytesIO()
    df.to_excel(output, index=False, sheet_name='Unmatched')
    return output.getvalue()

def _c(x): 
    try: return float(x.strip('%'))
    except: return 0.0

def generate_summary_report(stats: Dict, matched: List[Dict]) -> str:
    """Generate summary report"""
    report_lines = [
        "=" * 60,
        "APPOINTMENT PROCESSING SUMMARY REPORT",
        "=" * 60,
        "",
        f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL STATISTICS:",
        f"  Total Appointments Processed: {stats['total_appointments']}",
        f"  Total Names Extracted: {stats['total_names_extracted']}",
        f"  Entries Split (multiple patients): {stats['split_entries']}",
        f"  Metadata Extracted (CPT/TMS): {stats['metadata_extracted']}",
        "",
        "MATCHING RESULTS:",
        f"  Successfully Matched Patients: {stats['matched_patients']}",
        f"  Unmatched Entries: {stats['unmatched_entries']}",
        f"  Rejected (Low Confidence <75%): {stats.get('rejected_low_confidence', 0)}",
        f"  Fuzzy Matches: {stats['fuzzy_matches']}",
        f"  Ambiguous Matches: {stats['ambiguous_matches']}",
        "",
    ]
    
    if stats['total_names_extracted'] > 0:
        match_rate = stats['matched_patients']/stats['total_names_extracted']*100
        report_lines.append(f"  Match Rate: {match_rate:.1f}%")
    
    report_lines.append("")
    
    if matched:
        # Group by day
        by_day = {}
        for m in matched:
            day = m['day_of_week']
            by_day[day] = by_day.get(day, 0) + 1
        
        report_lines.append("APPOINTMENTS BY DAY:")
        for day, count in sorted(by_day.items()):
            report_lines.append(f"  {day}: {count}")
        report_lines.append("")
        
        # Group by doctor
        by_doctor = {}
        for m in matched:
            doc = m['doctor'] if m['doctor'] else "Unknown"
            by_doctor[doc] = by_doctor.get(doc, 0) + 1
        
        report_lines.append("APPOINTMENTS BY DOCTOR:")
        for doc, count in sorted(by_doctor.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {doc}: {count}")
        report_lines.append("")
        
        # Show CPT/TMS statistics
        with_codes = sum(1 for m in matched if m.get('codes'))
        if with_codes > 0:
            report_lines.append("CODES/PROCEDURES:")
            report_lines.append(f"  Appointments with CPT/TMS codes: {with_codes}")
            report_lines.append("")
        
        # Confidence distribution
        report_lines.append("CONFIDENCE DISTRIBUTION:")
        perfect = sum(1 for m in matched if _c(m['confidence']) == 100.0)
        high = sum(1 for m in matched if 90.0 <= _c(m['confidence']) < 100.0)
        medium = sum(1 for m in matched if 80.0 <= _c(m['confidence']) < 90.0)
        acceptable = sum(1 for m in matched if 75.0 <= _c(m['confidence']) < 80.0)
        report_lines.append(f"  Perfect matches (100%): {perfect}")
        report_lines.append(f"  High confidence (90-99%): {high}")
        report_lines.append(f"  Medium confidence (80-89%): {medium}")
        report_lines.append(f"  Acceptable confidence (75-79%): {acceptable}")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)

def create_log_content(log_messages: List[str]) -> str:
    """Create log file content"""
    current_date = datetime.now().strftime("%Y%m%d")
    log_content = f"Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += "=" * 60 + "\n\n"
    log_content += "\n".join(log_messages)
    return log_content

def create_zip_file(files_dict: Dict[str, bytes]) -> bytes:
    """Create a zip file containing all the output files"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            zip_file.writestr(filename, content)
    
    return zip_buffer.getvalue()

# Main Streamlit App
def main():
    st.title("📅 iCalendar Appointment Processor")
    st.markdown("Extract and process Monday/Friday appointments with patient matching")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Year selection
        current_year = datetime.now().year
        year = st.number_input(
            "Select Year",
            min_value=2020,
            max_value=2030,
            value=2025,
            help="Select the year to process appointments for"
        )
        
        # Month selection
        month_options = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        
        selected_month_names = st.multiselect(
            "Select Months",
            options=list(month_options.keys()),
            default=["September", "October"],
            help="Select one or more months to process"
        )
        
        selected_months = [month_options[month] for month in selected_month_names]
        
        st.markdown("---")
        st.info("📌 This app extracts Monday and Friday appointments from your calendar and matches them with patient records.")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload Files")
        
        # iCalendar file upload
        ical_file = st.file_uploader(
            "Upload iCalendar File (.ics)",
            type=['ics'],
            help="Upload your calendar file in iCalendar format"
        )
        
        # Patient list upload
        patient_file = st.file_uploader(
            "Upload Patient List (list_of_patients_mutual.xlsx)",
            type=['xlsx', 'xls'],
            help="Upload the Excel file containing patient information"
        )
    
    with col2:
        st.header("📊 Selected Parameters")
        st.write(f"**Year:** {year}")
        st.write(f"**Months:** {', '.join(selected_month_names)}")
        st.write(f"**Days:** Monday and Friday only")
    
    # Process button
    if st.button("🚀 Process Appointments", type="primary", use_container_width=True):
        if not ical_file:
            st.error("Please upload an iCalendar file")
        elif not patient_file:
            st.error("Please upload the patient list Excel file")
        elif not selected_months:
            st.error("Please select at least one month")
        else:
            with st.spinner("Processing appointments..."):
                log_messages = []
                try:
                    # Step 1: Parse iCalendar
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting iCalendar processing...")
                    calendar = parse_ical_file(ical_file.read())
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully parsed iCalendar file")
                    
                    # Step 2: Extract appointments
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting Monday/Friday appointments for {', '.join(selected_month_names)} {year}")
                    appointments = extract_appointments(calendar, selected_months, year)
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Extracted {len(appointments)} appointments")
                    
                    if not appointments:
                        st.warning("No appointments found for the specified criteria")
                        return
                    
                    # Step 3: Load patient list
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading patient list...")
                    patients = load_patient_list(patient_file.read())
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(patients)} patient records")
                    
                    # Step 4: Process and match
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing appointments and matching patients...")
                    matched, unmatched, stats = process_appointments_with_patients(appointments, patients)
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing complete: {stats['matched_patients']} matched, {stats['unmatched_entries']} unmatched")
                    
                    # Step 5: Generate output files
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generating output files...")
                    
                    # Create file contents
                    csv_content = create_csv_content(matched)
                    excel_content = create_excel_content(unmatched)
                    summary_content = generate_summary_report(stats, matched)
                    log_content = create_log_content(log_messages)
                    
                    # Store files in session state
                    current_date = datetime.now().strftime("%Y%m%d")
                    st.session_state.files_to_download = {
                        f"appointments_processed_{current_date}.csv": csv_content.encode('utf-8'),
                        f"appointments_not_patients_{current_date}.xlsx": excel_content,
                        f"processing_summary_{current_date}.txt": summary_content.encode('utf-8'),
                        f"run_log_{current_date}.txt": log_content.encode('utf-8')
                    }
                    
                    st.session_state.processed = True
                    st.session_state.stats = stats
                    st.session_state.summary = summary_content
                    
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] All files generated successfully!")
                    
                    st.success("✅ Processing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}")
    
    # Display results if processed
    if st.session_state.processed:
        st.markdown("---")
        st.header("📈 Processing Results")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Appointments", st.session_state.stats['total_appointments'])
        with col2:
            st.metric("Matched Patients", st.session_state.stats['matched_patients'])
        with col3:
            st.metric("Unmatched Entries", st.session_state.stats['unmatched_entries'])
        with col4:
            match_rate = (st.session_state.stats['matched_patients'] / 
                         st.session_state.stats['total_names_extracted'] * 100 
                         if st.session_state.stats['total_names_extracted'] > 0 else 0)
            st.metric("Match Rate", f"{match_rate:.1f}%")
        
        # Display summary report
        with st.expander("📋 View Summary Report"):
            st.text(st.session_state.summary)
        
        # Download section
        st.header("⬇️ Download Results")
        
        # Create zip file
        zip_content = create_zip_file(st.session_state.files_to_download)
        
        current_date = datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="📦 Download All Files (ZIP)",
            data=zip_content,
            file_name=f"appointment_processing_{current_date}.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        # Individual file downloads
        with st.expander("Download Individual Files"):
            col1, col2 = st.columns(2)
            
            with col1:
                for filename, content in list(st.session_state.files_to_download.items())[:2]:
                    if content:
                        mime = "text/csv" if filename.endswith('.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(
                            label=f"📄 {filename}",
                            data=content,
                            file_name=filename,
                            mime=mime
                        )
            
            with col2:
                for filename, content in list(st.session_state.files_to_download.items())[2:]:
                    if content:
                        st.download_button(
                            label=f"📄 {filename}",
                            data=content,
                            file_name=filename,
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
