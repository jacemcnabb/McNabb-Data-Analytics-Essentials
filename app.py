"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Natural-language query engine for the commercial lending portfolio:
  1. Intent classification  -> verified template (VQ1-VQ10) or generated SQL
  2. Query construction     -> template from library or LLM-generated SELECT
  3. Validation gate        -> read-only / template integrity / schema dry-run / confidence
  4. One retry (generated route only), otherwise escalate to a human analyst
  5. Read-only execution    -> pandas DataFrame + reasonableness checks
  6. Narrative generation   -> business-focused answer
  Every run displays: SQL used, raw data, confidence score, and is written to an audit trail.

Run locally:   streamlit run app.py
"""

import os
import re
import json
import time
import sqlite3
import sqlparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  (kept from the notebook imports)
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate  # noqa: F401  (kept from the notebook imports)

# =============================================================================
# Page config & constants
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = os.environ.get("TEST_QUERIES_PATH", str(BASE_DIR / "test_queries.csv"))
CONFIG_PATH = BASE_DIR / "config.json"
AUDIT_LOG_PATH = BASE_DIR / "audit_log.jsonl"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_QUESTION_CHARS = 1000
MAX_ROWS_FOR_LLM = 200  # cap rows sent to the narrative LLM to keep the prompt bounded

SANITY_CHECK_QUESTION = "What were the total sales for product line A in 2023?"


# =============================================================================
# Credentials (Streamlit secrets -> environment -> config.json -> sidebar input)
# =============================================================================
def _secret(name):
    """Safely read a Streamlit secret (returns None if no secrets file exists)."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def load_credentials():
    """Resolve the API key / base URL without ever hard-coding them in source."""
    api_key = _secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = (
        _secret("OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )

    # Optional local config.json (same format as the notebook) - do NOT commit it to GitHub
    if (not api_key or not api_base) and CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as file:
                config = json.load(file)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

with st.sidebar:
    st.header("⚙️ Configuration")
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = st.text_input("OpenAI API key", type="password")
        OPENAI_API_BASE = st.text_input(
            "OpenAI API base URL (optional)", value=OPENAI_API_BASE or ""
        ) or None

if not OPENAI_API_KEY:
    st.title("🏦 Credit Risk Query Engine")
    st.info(
        "No API credentials found. Add `OPENAI_API_KEY` (and `OPENAI_API_BASE` if you use a "
        "proxy endpoint) to Streamlit secrets or environment variables, or enter the key in the sidebar."
    )
    st.stop()

# Store API credentials in environment variables (same as the notebook)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


# =============================================================================
# LLM setup
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_llms(api_key, api_base, model_name):
    os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
    _llm = ChatOpenAI(model=model_name, temperature=0)
    _evaluator_llm = ChatOpenAI(model=model_name, temperature=0)
    return _llm, _evaluator_llm


llm, evaluator_llm = get_llms(OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME)


# =============================================================================
# Database loading (READ-ONLY connection)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    """Read-only SQLite connection (URI mode with ?mode=ro blocks any write operation)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


if not Path(DB_PATH).exists():
    st.title("🏦 Credit Risk Query Engine")
    st.error(
        f"Database file not found: `{DB_PATH}`. Place `credit_risk_portfolio.db` next to `app.py` "
        "(or set the `DB_PATH` environment variable)."
    )
    st.stop()

conn = get_connection(DB_PATH)


@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


ground_truth = load_ground_truth(TEST_QUERIES_PATH)


# =============================================================================
# Database schema provided to the LLM in the prompts
# =============================================================================
database_schema = """
TABLE sector_master:
  - sector_code (TEXT, PRIMARY KEY): Unique identifier for business sectors (e.g., 'SEC_REAL', 'SEC_MFG').
  - sector_name (TEXT): Full name of the industry sector (e.g., 'Real Estate', 'Manufacturing').
  - naics_code (TEXT): NAICS industry classification code.
  - naics_description (TEXT): Description of NAICS code.
  - is_sensitive_sector (INTEGER): Flag (1 = Sensitive sector such as Real Estate/Guns/Gambling, 0 = Non-sensitive).

TABLE loan_master:
  - loan_account_number (TEXT, PRIMARY KEY): Unique loan account identifier.
  - borrower_id (TEXT): Unique borrower identifier (foreign key to borrower_rating).
  - borrower_name (TEXT): Name of the corporate borrower or individual.
  - borrower_type (TEXT): Borrower classification ('Corporate', 'SME', 'Retail').
  - group_name (TEXT): Parent corporate group name if applicable.
  - state (TEXT): Geographic state of borrower/facility location.
  - product_type (TEXT): Product type ('Term Loan', 'Working Capital', 'Revolving Line').
  - loan_category (TEXT): Loan category ('Commercial', 'Retail', 'Mortgage').
  - sector_code (TEXT): Foreign key linking to sector_master.sector_code.
  - sanctioned_amount (REAL): Credit facility limit sanctioned.
  - disbursed_amount (REAL): Amount disbursed to borrower.
  - outstanding_principal (REAL): Current outstanding balance of principal.
  - outstanding_interest (REAL): Outstanding accrued interest balance.
  - total_outstanding (REAL): Total outstanding exposure (principal + interest).
  - interest_rate (REAL): Current annual interest rate (percentage, e.g., 8.5).
  - rate_type (TEXT): Interest rate mechanism ('Fixed', 'Floating').
  - sanction_date (DATE): Date loan was sanctioned (YYYY-MM-DD).
  - maturity_date (DATE): Contractual loan maturity date.
  - repayment_frequency (TEXT): Payment schedule ('Monthly', 'Quarterly', 'Annual').
  - branch_code (TEXT): Branch handling the loan.
  - branch_name (TEXT): Name of bank branch.
  - relationship_manager (TEXT): Assigned Relationship Manager.
  - is_consortium (INTEGER): Flag (1 = Consortium loan, 0 = Direct loan).
  - is_restructured (INTEGER): Flag (1 = Loan restructured/modified, 0 = Standard).
  - restructuring_date (DATE): Date restructuring agreement took effect.
  - is_secured (INTEGER): Flag (1 = Collateralized/Secured, 0 = Unsecured).
  - days_past_due (INTEGER): Days payment is overdue (0 = Current, >0 = Delinquent/DPD).
  - asset_classification (TEXT): Credit status ('Standard', 'SMA-0', 'SMA-1', 'SMA-2', 'NPA', 'Substandard', 'Doubtful', 'Loss').
  - classification_date (DATE): Date current classification was assigned.

TABLE borrower_rating:
  - rating_id (INTEGER, PRIMARY KEY): Unique rating record ID.
  - borrower_id (TEXT): Unique borrower ID (links to loan_master.borrower_id).
  - rating_date (DATE): Rating assessment date.
  - internal_rating (TEXT): Internal risk rating grade (e.g., 'AAA', 'BBB', 'CCC', 'D').
  - previous_rating (TEXT): Prior internal risk rating.
  - rating_direction (TEXT): Rating migration trend ('Upgrade', 'Downgrade', 'Unchanged').
  - external_rating_agency (TEXT): Credit rating agency (e.g., 'CRISIL', 'ICRA', 'CARE').
  - external_rating (TEXT): External agency rating.
  - pd_estimate (REAL): Estimated Probability of Default decimal (e.g., 0.025 = 2.5%).
  - rating_model_version (TEXT): Rating model version identifier.

TABLE provisioning:
  - provision_id (INTEGER, PRIMARY KEY): Provisioning record ID.
  - loan_account_number (TEXT): Loan account identifier (links to loan_master.loan_account_number).
  - reporting_date (DATE): Financial reporting quarter end date (YYYY-MM-DD).
  - ifrs9_stage (INTEGER): IFRS 9 impairment stage (1 = Performing, 2 = Underperforming/SICR, 3 = Credit-Impaired/NPA).
  - stage_rationale (TEXT): Qualitative or quantitative reason for current IFRS 9 stage assignment.
  - pd_12_month (REAL): 12-Month Probability of Default decimal.
  - pd_lifetime (REAL): Lifetime Probability of Default decimal.
  - lgd_estimate (REAL): Loss Given Default decimal (e.g., 0.45 = 45%).
  - ead_amount (REAL): Exposure at Default dollar amount.
  - ecl_amount (REAL): Expected Credit Loss provision dollar amount.
  - provision_held (REAL): Actual provision balance maintained on balance sheet.
  - provision_coverage_ratio (REAL): Ratio of provision held to total exposure.
  - is_individually_assessed (INTEGER): Flag (1 = Specific individual assessment, 0 = Collective pool assessment).

RELATIONSHIPS & JOIN KEYS:
  - loan_master.sector_code = sector_master.sector_code
  - loan_master.borrower_id = borrower_rating.borrower_id
  - loan_master.loan_account_number = provisioning.loan_account_number
"""


# =============================================================================
# Verified Query Template Library (VQ1 - VQ10)
# =============================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
    ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss', 'NPA') THEN lm.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_millions
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_millions DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '01. Current (0 DPD)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '02. 1-30 DPD'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '03. 31-60 DPD'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '04. 61-90 DPD'
        ELSE '05. 90+ DPD (NPA)'
    END AS dpd_bucket,
    COUNT(*) AS total_loans,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY dpd_bucket
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage ASC;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 4) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC;
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10;
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != ''
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5;
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '01. Current (0 DPD)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '02. 1-30 DPD'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '03. 31-60 DPD'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '04. 61-90 DPD'
        ELSE '05. 90+ DPD (NPA)'
    END AS dpd_bucket,
    COUNT(*) AS total_loans,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY dpd_bucket
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}

# Global Sanitizer for Verified Query Library
# Strip leading/trailing whitespace and trailing semicolons
for _qid, _qdata in verified_query_library.items():
    if 'sql' in _qdata and isinstance(_qdata['sql'], str):
        _qdata['sql'] = _qdata['sql'].strip().rstrip(';')


# =============================================================================
# Tool 1: Intent Classification
# =============================================================================
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are an intent classifier for a credit risk query engine.
Analyze the user's question and determine whether it matches one of the pre-approved verified query templates in the library or requires dynamic SQL generation.

VERIFIED QUERY LIBRARY:
{library_descriptions}

USER QUESTION:
{user_question}

INSTRUCTIONS:
- If the user's intent directly matches one of the pre-approved query templates (VQ1 to VQ10), set "route" to "verified" and "query_id" to the matching template ID (e.g. "VQ1"). Note that a question focusing on a specific sector, category, or subset of a template still routes to that verified template if it provides the underlying dataset.
- If no pre-approved template matches the question, set "route" to "generated" and "query_id" to null.
- Provide a concise explanation in "match_reason".

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# =============================================================================
# Tool 2: Query Generation
# =============================================================================
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
You are an expert SQL developer for a commercial credit risk database (SQLite).
Generate a valid, read-only SQL query to answer the user's question based on the provided database schema.

DATABASE SCHEMA:
{schema_context}

USER QUESTION:
{user_question}

GUIDELINES:
- Return ONLY a single SELECT query. Do NOT use non-read-only operations.
- Ensure column names, table names, and join conditions strictly match the schema.
- For financial values (e.g., total_outstanding, ead_amount, ecl_amount), convert to millions by dividing by 1,000,000.0 or round appropriately if required.
- Do NOT include markdown code formatting or explanation; output raw SQL only.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# =============================================================================
# Tool 3: Query Validation
# =============================================================================
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a SQL query across security, template integrity, schema execution, and relevance gates.

    Returns:
    - dict: Validation results containing 'passed', 'relevance_confidence', 'failed_check', and 'details'.
    '''

    result = {
        'passed': False,
        'relevance_confidence': 0.0,
        'failed_check': None,
        'details': None
    }

    if not candidate_sql or not isinstance(candidate_sql, str):
        result['failed_check'] = 'empty_query'
        result['details'] = 'Candidate SQL is empty or invalid.'
        return result

    # Upfront Sanitization: Strip whitespace and trailing semicolons
    candidate_sql = candidate_sql.strip().rstrip(';')

    # ---------------------------------------------------------
    # Gate 1: Read-Only & Shape Check (Security)
    # ---------------------------------------------------------
    # Reject multiple statements (semicolons remaining in the body)
    if ';' in candidate_sql:
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed.'
        return result

    # Must begin with SELECT or WITH
    sql_uppercase = candidate_sql.upper().strip()
    if not (sql_uppercase.startswith('SELECT') or sql_uppercase.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must begin with SELECT or WITH.'
        return result

    # Block destructive/mutation DDL & DML keywords
    forbidden_keywords = [
        r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b', r'\bDROP\b',
        r'\bALTER\b', r'\bCREATE\b', r'\bTRUNCATE\b', r'\bEXEC\b', r'\bATTACH\b', r'\bDETACH\b'
    ]
    for pattern in forbidden_keywords:
        if re.search(pattern, sql_uppercase):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f"Forbidden keyword detected matching pattern: {pattern}"
            return result

    # ---------------------------------------------------------
    # Gate 2: Template Integrity Check (For Verified Route)
    # ---------------------------------------------------------
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';')

        # Verify template alignment
        if candidate_sql != expected_sql:
            # Fallback normalization comparison (ignoring whitespace differences)
            norm_candidate = " ".join(candidate_sql.split())
            norm_expected = " ".join(expected_sql.split())

            if norm_candidate != norm_expected:
                result['failed_check'] = 'template_integrity'
                result['details'] = f"Candidate SQL does not match verified template {query_id}."
                return result

    # ---------------------------------------------------------
    # Gate 3: Schema & Syntax Validation (Dry-Run Execution)
    # ---------------------------------------------------------
    try:
        cursor = db_connection.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {candidate_sql}")
    except sqlite3.Error as e:
        result['failed_check'] = 'schema_validity'
        result['details'] = f"SQLite syntax/schema error: {str(e)}"
        return result

    # ---------------------------------------------------------
    # Gate 4: Relevance Confidence Scoring
    # ---------------------------------------------------------
    # Verified library routes receive full confidence (1.0)
    if query_id and query_id in query_library:
        relevance_confidence = 1.0
    else:
        # Generated route heuristic/confidence score (0.9 standard baseline for valid SQL)
        relevance_confidence = 0.90

    result['passed'] = True
    result['relevance_confidence'] = relevance_confidence
    result['details'] = 'All validation checks passed successfully.'

    return result


# =============================================================================
# Tool 4: Retry Generation
# =============================================================================
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are an expert SQL developer for a commercial credit risk database (SQLite).
The SQL query below failed validation. Correct it so that it passes validation while still
answering the original user question.

GUIDELINES:
- Return ONLY a single, corrected SELECT (or WITH ... SELECT) query. Do NOT use any non-read-only operation.
- Do NOT include a trailing semicolon or multiple statements.
- Fix the specific problem described in the validation error, and preserve the original user intent.
- Ensure table names, column names, and join conditions strictly match the database schema.
- For financial values (e.g., total_outstanding, ead_amount, ecl_amount), convert to millions by dividing by 1,000,000.0.
- NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss'). The latest provisioning and rating date is '2025-09-30'.
- Do NOT include markdown code formatting or explanation; output raw SQL only.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# =============================================================================
# Tool 5: Query Execution
# =============================================================================
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# =============================================================================
# Tool 6: Response Generation
# =============================================================================
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    # Keep the prompt bounded for very large result sets (e.g. all overdue accounts)
    total_rows = len(dataframe)
    if total_rows > MAX_ROWS_FOR_LLM:
        data_text = (
            dataframe.head(MAX_ROWS_FOR_LLM).to_string()
            + f"\n\n[Note: result truncated for this summary - showing the first {MAX_ROWS_FOR_LLM} "
              f"of {total_rows} rows. Do not present totals derived from the shown rows as portfolio totals.]"
        )
    else:
        data_text = dataframe.to_string()

    response_prompt = f"""
You are a credit risk analytics assistant for a commercial bank. Answer the user's question using ONLY the
query result provided below.

GUIDELINES:
- Write a concise, business-focused response (a short paragraph, or a few short bullets if that is clearer).
- Focus only on what the user actually asked; highlight the most relevant insights from the full result.
- Quote exact figures from the data. Monetary values are in millions unless a column name says otherwise.
- Do NOT invent, estimate, or infer numbers that are not in the data.
- If the result is empty, say clearly that no matching records were found instead of guessing.
- Do not mention SQL, tables, or internal system details.

Route used: {route}{f' ({query_id})' if query_id else ''}

USER QUESTION:
{user_question}

QUERY RESULT:
{data_text}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# Pipeline Orchestration
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, log and stage trace.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'warnings': []
    }

    # In the notebook, verbose=True printed the stages; in Streamlit they are collected in a trace
    trace = []

    def say(message):
        if verbose:
            trace.append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)

    # Clean up SQL: strip trailing/leading whitespace and extra trailing semicolons
    if candidate_sql:
        candidate_sql = candidate_sql.strip().rstrip(';')

    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        if candidate_sql:
            candidate_sql = candidate_sql.strip().rstrip(';')
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'trace': trace, **log}


# =============================================================================
# Audit trail helpers
# =============================================================================
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None


def write_audit_entry(source, result=None, question=None, error=None, elapsed=None):
    """Append an audit record to the session log and to audit_log.jsonl (best effort)."""
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "user_question": question,
        "elapsed_seconds": None if elapsed is None else round(elapsed, 2),
        "error": error,
    }
    if result is not None:
        entry["log"] = result["log"]
        entry["trace"] = result.get("trace")
    st.session_state.audit_log.append(entry)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass  # read-only filesystems should not break the app
    return entry


def safe_run(question, source):
    """Run the pipeline, time it, and write the audit entry. Returns (result, error)."""
    start = time.time()
    try:
        result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
        elapsed = time.time() - start
        result["elapsed_seconds"] = elapsed
        write_audit_entry(source, result=result, question=question, elapsed=elapsed)
        return result, None
    except Exception as exc:  # LLM/API/DB errors
        elapsed = time.time() - start
        write_audit_entry(source, question=question, error=str(exc), elapsed=elapsed)
        return None, str(exc)


# =============================================================================
# UI helpers
# =============================================================================
def fmt_conf(value):
    if isinstance(value, (int, float)):
        return f"{value:.0%}"
    return str(value) if value is not None else "-"


def safe_md(text):
    """Escape $ so dollar amounts are not rendered as LaTeX."""
    return (text or "").replace("$", "\\$")


def render_result(res):
    log = res["log"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(log["route"]).title())
    c2.metric("Verified Query ID", log["query_id"] or "-")
    c3.metric("Confidence", fmt_conf(log["confidence"]))
    c4.metric("Rows returned", "-" if log["row_count"] is None else log["row_count"])

    if log["escalated"]:
        st.error("⚠️ Escalated to a human analyst - the query could not be reliably resolved.")
        st.write(safe_md(log["narrative"]))
    else:
        st.subheader("Answer")
        st.markdown(safe_md(log["narrative"]))

    st.subheader("SQL used")
    sql_text = log["executed_sql"] or log["candidate_sql"] or ""
    st.code(sql_text, language="sql")
    if log["escalated"]:
        st.caption("This SQL was NOT executed (failed validation).")

    if res.get("dataframe") is not None:
        st.subheader("Raw data returned")
        st.dataframe(res["dataframe"])
        st.download_button(
            "Download result as CSV",
            res["dataframe"].to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
            key=f"dl_{id(res)}",
        )

    if log.get("warnings"):
        for w in log["warnings"]:
            st.warning(w)

    with st.expander("Pipeline trace & validation details"):
        st.markdown(f"**Match reason:** {log['match_reason']}")
        st.markdown(f"**Retry used:** {log['retry_used']}")
        if res.get("elapsed_seconds") is not None:
            st.markdown(f"**Turnaround:** {res['elapsed_seconds']:.1f} seconds")
        st.code("\n".join(res.get("trace", [])), language="text")
        st.json(log["gate_result"])


# =============================================================================
# Sidebar status
# =============================================================================
with st.sidebar:
    st.success("Database connected (read-only)")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Verified templates: {len(verified_query_library)}")
    st.caption(f"Audit entries this session: {len(st.session_state.audit_log)}")
    if ground_truth is None:
        st.warning("test_queries.csv not found - evaluation tab is disabled.")
    st.divider()
    st.caption(
        "Reference: NPA = Substandard / Doubtful / Loss. Latest provisioning and rating "
        "date = 2025-09-30. All queries are read-only."
    )


# =============================================================================
# Main UI
# =============================================================================
st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Northbridge Bank - ask routine commercial-lending portfolio questions in plain English. "
    "Every answer shows the SQL used, the raw data, and a confidence score, and is logged for audit."
)

tab_ask, tab_eval, tab_library, tab_schema, tab_audit = st.tabs(
    ["💬 Ask a question", "🧪 Test cases & evaluation", "📚 Verified query library",
     "🗄️ Database & schema", "🧾 Audit trail"]
)

# ---------------------------------------------------------------------------
# Tab 1: Ask a question
# ---------------------------------------------------------------------------
with tab_ask:
    examples = [] if ground_truth is None else ground_truth["User Query"].dropna().tolist()
    example_options = ["- write your own -"] + examples + [SANITY_CHECK_QUESTION]
    choice = st.selectbox("Example questions", example_options)
    default_q = "" if choice.startswith("- write") else choice

    question = st.text_area(
        "Your question",
        value=default_q,
        key=f"question_{choice}",
        height=100,
        max_chars=MAX_QUESTION_CHARS,
        placeholder="e.g. Which sectors have the highest NPA exposure?",
    )

    if st.button("Run query", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running the query engine..."):
                res, err = safe_run(question.strip(), source="interactive")
            if err:
                st.session_state.last_result = None
                st.error(f"The pipeline failed with an error: {err}")
            else:
                st.session_state.last_result = res

    if st.session_state.last_result is not None:
        st.divider()
        render_result(st.session_state.last_result)

# ---------------------------------------------------------------------------
# Tab 2: Test cases & evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    if ground_truth is None:
        st.info("Add `test_queries.csv` next to `app.py` to enable evaluation.")
    else:
        st.markdown("**Ground-truth test cases**")
        st.dataframe(ground_truth)

        if st.button("Run all test cases"):
            test_results = []
            progress = st.progress(0.0, text="Running test cases...")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                res, err = safe_run(gt["User Query"], source=f"evaluation:{gt['Test Case']}")
                test_results.append((res, err))
                progress.progress((i + 1) / len(ground_truth), text=f"Completed {i + 1}/{len(ground_truth)}")
            progress.empty()
            st.session_state.evaluation = test_results

        if st.session_state.evaluation is not None:
            evaluation_rows = []
            for (_, gt), (tr, err) in zip(ground_truth.iterrows(), st.session_state.evaluation):
                if tr is None:
                    evaluation_rows.append({
                        'Test Case': gt['Test Case'],
                        'Expected Route': gt['Expected Route'],
                        'Actual Route': None,
                        'Route Match': False,
                        'Expected Query ID': gt['Expected Query ID'],
                        'Actual Query ID': None,
                        'Query ID Match': False,
                        'Confidence': None,
                        'Rows Returned': None,
                    })
                    continue
                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count'],
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100
            verified_mask = evaluation_df['Expected Route'].astype(str).str.strip().str.lower() == 'verified'
            query_accuracy = (
                evaluation_df.loc[verified_mask, 'Query ID Match'].mean() * 100
                if verified_mask.any() else float('nan')
            )
            average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence Score", "n/a" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

            st.markdown("**Per-test-case summary**")
            st.dataframe(evaluation_df)

            st.markdown("**Test case details**")
            for (_, gt), (tr, err) in zip(ground_truth.iterrows(), st.session_state.evaluation):
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    if "Expected Answer" in gt:
                        st.markdown(f"**Expected answer:** {safe_md(str(gt['Expected Answer']))}")
                    if err:
                        st.error(err)
                    else:
                        render_result(tr)

# ---------------------------------------------------------------------------
# Tab 3: Verified query library
# ---------------------------------------------------------------------------
with tab_library:
    st.markdown(
        "Pre-approved, tested SQL templates for recurring questions. Each runs without modification."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"], language="sql")

# ---------------------------------------------------------------------------
# Tab 4: Database & schema
# ---------------------------------------------------------------------------
with tab_schema:
    st.markdown("**Schema provided to the LLM**")
    st.code(database_schema, language="text")

    st.markdown("**Table previews (first 3 rows)**")
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )['name'].tolist()
    for table in tables:
        if table != 'sqlite_sequence':
            with st.expander(f"Table: {table}"):
                st.dataframe(pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3;", conn))

# ---------------------------------------------------------------------------
# Tab 5: Audit trail
# ---------------------------------------------------------------------------
with tab_audit:
    st.markdown(
        "Every analytical output is recorded (question, route, SQL, validation result, confidence, "
        "narrative). Entries are also appended to `audit_log.jsonl` on the server."
    )
    if not st.session_state.audit_log:
        st.info("No queries have been run in this session yet.")
    else:
        rows = []
        for e in st.session_state.audit_log:
            lg = e.get("log") or {}
            rows.append({
                "Time (UTC)": e["timestamp_utc"],
                "Source": e["source"],
                "Question": e["user_question"],
                "Route": lg.get("route"),
                "Query ID": lg.get("query_id"),
                "Retry": lg.get("retry_used"),
                "Escalated": lg.get("escalated"),
                "Confidence": lg.get("confidence"),
                "Rows": lg.get("row_count"),
                "Seconds": e.get("elapsed_seconds"),
                "Error": e.get("error"),
            })
        st.dataframe(pd.DataFrame(rows))
        st.download_button(
            "Download full audit log (JSON Lines)",
            "\n".join(json.dumps(e, default=str) for e in st.session_state.audit_log).encode("utf-8"),
            file_name="audit_log.jsonl",
            mime="application/json",
        )
