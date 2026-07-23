/**
 * FN Risk Prediction System — Main Application Component
 *
 * This React application provides the clinical decision-support interface
 * for a LightGBM-based febrile neutropenia (FN) risk prediction model.
 * Clinicians enter patient demographics, vital signs, laboratory values,
 * chemotherapy drug doses (by YJ code), and ICD-10 cancer diagnoses.
 * The data are submitted to the FastAPI backend, which returns a
 * probabilistic FN risk score and binary risk classification.
 *
 * Component hierarchy
 * -------------------
 *   App          — Root component; owns all state and API communication.
 *   ├── FField   — Labelled form-field wrapper with optional unit display.
 *   ├── ResultPanel  — Displays the model prediction output.
 *   └── EmptyResult  — Placeholder shown before the first prediction.
 */

import { useState, useMemo } from 'react'
import './App.css'

/** Base URL for the FastAPI prediction backend. */
const API_BASE_URL = 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Reference data: YJ pharmaceutical classification codes
// These 208 codes correspond to Column_10 – Column_217 of the feature matrix
// and represent the chemotherapy and supportive-care drugs present in the
// training dataset.
// ---------------------------------------------------------------------------
const YJ_CODES = [
  "1190402","1214405","1317715","1319401","2399402","2399403","2399404",
  "2454002","2454405","2478002","2499406","2499407","2499412","3919401",
  "3929407","3929410","3999423","3999429","3999445","3999462","4211002",
  "4211401","4211402","4212400","4213400","4219002","4219003","4219004",
  "4219400","4219401","4219402","4219403","4219404","4219405","4219406",
  "4221001","4222001","4222400","4223004","4223005","4223401","4223701",
  "4224401","4224403","4229001","4229100","4229101","4229400","4229401",
  "4229402","4229403","4231400","4233400","4234400","4234402","4235400",
  "4235401","4235402","4235403","4235404","4235405","4235406","4239400",
  "4239401","4240001","4240400","4240401","4240402","4240403","4240404",
  "4240405","4240406","4240407","4240408","4240409","4240410","4291002",
  "4291003","4291005","4291006","4291007","4291009","4291010","4291011",
  "4291012","4291013","4291015","4291016","4291017","4291018","4291019",
  "4291020","4291021","4291023","4291024","4291026","4291027","4291028",
  "4291029","4291030","4291031","4291032","4291033","4291034","4291036",
  "4291037","4291038","4291039","4291040","4291043","4291044","4291045",
  "4291046","4291047","4291048","4291049","4291051","4291052","4291053",
  "4291054","4291055","4291057","4291058","4291059","4291062","4291063",
  "4291064","4291065","4291066","4291068","4291069","4291070","4291075",
  "4291076","4291077","4291400","4291401","4291402","4291403","4291405",
  "4291406","4291407","4291408","4291409","4291410","4291412","4291413",
  "4291415","4291416","4291417","4291419","4291420","4291421","4291422",
  "4291424","4291425","4291426","4291427","4291428","4291429","4291430",
  "4291431","4291433","4291434","4291435","4291436","4291437","4291438",
  "4291439","4291441","4291442","4291443","4291444","4291445","4291446",
  "4291448","4291449","4291450","4291452","4291454","4291455","4291457",
  "4291459","4291464","4291465","4291466","4291468","4291469","4291471",
  "4291500","4291501","4299002","4299003","4299100","4299400","4299404",
  "4299406","4900402","4900404","6391700","6399402","6399413","6399421",
  "6399423","6399424","6399425","6399427","6399429",
]

// ---------------------------------------------------------------------------
// Reference data: ICD-10 Chapter C (malignant neoplasm) diagnosis codes
// These 68 codes correspond to Column_220 – Column_287 of the feature matrix.
// ---------------------------------------------------------------------------
const DIAGNOSIS_CODES = [
  "C",  "C00","C01","C02","C03","C04","C05","C06","C07","C08","C09",
  "C10","C11","C12","C13","C15","C16","C17","C18","C19","C20","C21",
  "C22","C23","C24","C25","C26","C30","C31","C32","C33","C34","C37",
  "C38","C40","C41","C42","C44","C47","C48","C49","C50","C51","C52",
  "C53","C54","C56","C57","C58","C60","C61","C62","C63","C64","C65",
  "C66","C67","C68","C69","C70","C71","C72","C73","C74","C75","C76",
  "C77","C80",
]

/**
 * Human-readable English labels for each ICD-10 Chapter C diagnosis code,
 * used to render descriptive checkboxes in the diagnosis selection panel.
 */
const DIAGNOSIS_LABELS = {
  C:   "Malignant neoplasm (general)",
  C00: "Lip",                          C01: "Base of tongue",
  C02: "Tongue (other)",               C03: "Gum",
  C04: "Floor of mouth",               C05: "Palate",
  C06: "Mouth (other)",                C07: "Parotid gland",
  C08: "Major salivary gland",         C09: "Tonsil",
  C10: "Oropharynx",                   C11: "Nasopharynx",
  C12: "Pyriform sinus",               C13: "Hypopharynx",
  C15: "Esophagus",                    C16: "Stomach",
  C17: "Small intestine",              C18: "Colon",
  C19: "Rectosigmoid junction",        C20: "Rectum",
  C21: "Anus and anal canal",          C22: "Liver and intrahepatic bile ducts",
  C23: "Gallbladder",                  C24: "Biliary tract (other)",
  C25: "Pancreas",                     C26: "Digestive organs (other)",
  C30: "Nasal cavity and middle ear",  C31: "Accessory sinuses",
  C32: "Larynx",                       C33: "Trachea",
  C34: "Bronchus and lung",            C37: "Thymus",
  C38: "Heart, mediastinum and pleura",
  C40: "Bone and cartilage of limbs",  C41: "Bone and articular cartilage (other)",
  C42: "Haematopoietic and lymphoid",  C44: "Skin (other)",
  C47: "Peripheral nerves",            C48: "Retroperitoneum and peritoneum",
  C49: "Connective and soft tissue",   C50: "Breast",
  C51: "Vulva",                        C52: "Vagina",
  C53: "Cervix uteri",                 C54: "Corpus uteri",
  C56: "Ovary",                        C57: "Female genital organs (other)",
  C58: "Placenta",                     C60: "Penis",
  C61: "Prostate",                     C62: "Testis",
  C63: "Male genital organs (other)",  C64: "Kidney",
  C65: "Renal pelvis",                 C66: "Ureter",
  C67: "Bladder",                      C68: "Urinary organs (other)",
  C69: "Eye and adnexa",               C70: "Meninges",
  C71: "Brain",                        C72: "Spinal cord and cranial nerves",
  C73: "Thyroid gland",                C74: "Adrenal gland",
  C75: "Endocrine glands (other)",     C76: "Ill-defined sites",
  C77: "Lymph nodes (secondary)",      C80: "Malignant neoplasm, primary site unknown",
}

// ---------------------------------------------------------------------------
// Default form values — representative of a typical patient in the cohort
// ---------------------------------------------------------------------------
const defaultFormData = {
  LastNEUT: 1540, LastWBC: 4500, AGE: 65,
  MS_HEIGHT: 165, MS_WEIGHT: 62.5,
  TEMPR: 36.6, PULSE: 72, RESP: 16,
  BPH: 128, BPL: 82,
  PT_SEX: 1, TOTAL_GLAS_PREDOSE: 0,
  drugs: {}, diagnoses: [],
}

// ===========================================================================
// Root application component
// ===========================================================================
export default function App() {
  // Core application state
  const [loading, setLoading]   = useState(false)   // API request in progress
  const [result, setResult]     = useState(null)     // Latest prediction response
  const [formData, setFormData] = useState(defaultFormData)

  // Drug entry sub-form state
  const [drugSearch, setDrugSearch]       = useState('')
  const [drugInputCode, setDrugInputCode] = useState('')
  const [drugInputDose, setDrugInputDose] = useState('')

  // Diagnosis panel filter state
  const [diagSearch, setDiagSearch] = useState('')

  // Filtered YJ code list for the drug autocomplete dropdown
  const filteredYjCodes = useMemo(() =>
    drugSearch.trim() === '' ? YJ_CODES
      : YJ_CODES.filter(c => c.includes(drugSearch.trim())),
    [drugSearch]
  )

  // Filtered diagnosis code list for the diagnosis selection panel
  const filteredDiagCodes = useMemo(() =>
    diagSearch.trim() === '' ? DIAGNOSIS_CODES
      : DIAGNOSIS_CODES.filter(c =>
          c.toLowerCase().includes(diagSearch.trim().toLowerCase()) ||
          (DIAGNOSIS_LABELS[c] || '').toLowerCase().includes(diagSearch.trim().toLowerCase())
        ),
    [diagSearch]
  )

  /** Update a numeric form field from an input change event. */
  const handleChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : parseInt(value),
    }))
  }

  /** Add a drug entry to the form data after validating the code and dose. */
  const addDrug = () => {
    const code = drugInputCode.trim()
    const dose = parseFloat(drugInputDose)
    if (!code || isNaN(dose) || dose <= 0) return
    setFormData(prev => ({ ...prev, drugs: { ...prev.drugs, [code]: dose } }))
    setDrugInputCode('')
    setDrugInputDose('')
    setDrugSearch('')
  }

  /** Remove a previously added drug entry by YJ code. */
  const removeDrug = (code) => {
    setFormData(prev => {
      const d = { ...prev.drugs }; delete d[code]
      return { ...prev, drugs: d }
    })
  }

  /** Toggle the presence of an ICD-10 diagnosis code in the selection list. */
  const toggleDiag = (code) => {
    setFormData(prev => ({
      ...prev,
      diagnoses: prev.diagnoses.includes(code)
        ? prev.diagnoses.filter(c => c !== code)
        : [...prev.diagnoses, code],
    }))
  }

  /**
   * Populate the form with a pre-defined sample patient profile.
   * Intended to aid demonstration; not for clinical use.
   *
   * @param {'high'|'low'} type - Risk profile to load.
   */
  const fillSample = (type) => {
    if (type === 'high') {
      // Sample high-risk profile: elderly patient, severe neutropenia, fever
      setFormData(prev => ({ ...prev, LastNEUT: 450, LastWBC: 1800, AGE: 78,
        MS_WEIGHT: 45, TEMPR: 37.9, TOTAL_GLAS_PREDOSE: 0, diagnoses: ['C34'] }))
    } else {
      // Sample low-risk profile: younger patient, normal counts, G-CSF prophylaxis
      setFormData(prev => ({ ...prev, LastNEUT: 2500, LastWBC: 6000, AGE: 45,
        MS_WEIGHT: 70, TEMPR: 36.5, TOTAL_GLAS_PREDOSE: 300, diagnoses: ['C50'] }))
    }
  }

  /**
   * Submit the form data to the /predict endpoint and store the result.
   * Displays an alert if the API is unreachable or returns an error.
   */
  const handlePredict = async (e) => {
    e.preventDefault()
    setLoading(true); setResult(null)
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })
      if (!res.ok) throw new Error('API Error')
      setResult(await res.json())
    } catch (err) {
      console.error(err)
      alert('Prediction failed. Please verify that the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  // Show the YJ code autocomplete dropdown only when the search field contains
  // a partial, unresolved query.
  const showDropdown = drugSearch.trim() !== ''
    && filteredYjCodes.length > 0
    && !YJ_CODES.includes(drugSearch.trim())

  return (
    <div className="app-container">
      {/* Full-screen loading overlay displayed during API inference */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <p className="loading-text">Analysing...</p>
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Page header                                                       */}
      {/* ---------------------------------------------------------------- */}
      <header>
        <div>
          <h1>FN Risk Prediction System</h1>
          <p className="subtitle">Febrile Neutropenia (FN) Risk Assessment</p>
        </div>
        <span className="portal-badge">For Clinicians</span>
      </header>

      <main className="main-content">
        {/* ============================================================== */}
        {/* Left panel: Patient data entry form                            */}
        {/* ============================================================== */}
        <section className="input-section">

          {/* Sample data quick-fill buttons */}
          <div className="sample-bar">
            <span className="sample-label">Sample data:</span>
            <button type="button" className="btn btn-secondary" onClick={() => fillSample('low')}>Low-Risk Example</button>
            <button type="button" className="btn btn-secondary" onClick={() => fillSample('high')}>High-Risk Example</button>
          </div>

          <form onSubmit={handlePredict}>

            {/* ---------------------------------------------------------- */}
            {/* Section 1: Demographics and vital signs                    */}
            {/* ---------------------------------------------------------- */}
            <div className="section-block">
              <p className="section-title">Demographics / Vital Signs</p>
              <div className="grid-3">
                <FField label="Age" unit="years">
                  <input type="number" name="AGE" value={formData.AGE} onChange={handleChange} />
                </FField>
                <FField label="Sex">
                  <select name="PT_SEX" value={formData.PT_SEX} onChange={handleChange}>
                    <option value={0}>Female</option>
                    <option value={1}>Male</option>
                  </select>
                </FField>
                <FField label="Height" unit="cm">
                  <input type="number" step="0.1" name="MS_HEIGHT" value={formData.MS_HEIGHT} onChange={handleChange} />
                </FField>
                <FField label="Weight" unit="kg">
                  <input type="number" step="0.1" name="MS_WEIGHT" value={formData.MS_WEIGHT} onChange={handleChange} />
                </FField>
                <FField label="Temperature" unit="°C">
                  <input type="number" step="0.1" name="TEMPR" value={formData.TEMPR} onChange={handleChange} />
                </FField>
                <FField label="Heart Rate" unit="bpm">
                  <input type="number" name="PULSE" value={formData.PULSE} onChange={handleChange} />
                </FField>
                <FField label="Respiratory Rate" unit="breaths/min">
                  <input type="number" name="RESP" value={formData.RESP} onChange={handleChange} />
                </FField>
                <FField label="Systolic BP" unit="mmHg">
                  <input type="number" name="BPH" value={formData.BPH} onChange={handleChange} />
                </FField>
                <FField label="Diastolic BP" unit="mmHg">
                  <input type="number" name="BPL" value={formData.BPL} onChange={handleChange} />
                </FField>
              </div>
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Section 2: Laboratory values (most recent)                 */}
            {/* ---------------------------------------------------------- */}
            <div className="section-block">
              <p className="section-title">Laboratory Values (Most Recent)</p>
              <div className="grid-2">
                <FField label="Neutrophil Count (ANC)" unit="/μL">
                  <input type="number" name="LastNEUT" value={formData.LastNEUT} onChange={handleChange} />
                </FField>
                <FField label="White Blood Cell Count (WBC)" unit="/μL">
                  <input type="number" name="LastWBC" value={formData.LastWBC} onChange={handleChange} />
                </FField>
              </div>
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Section 3: Prophylactic G-CSF administration               */}
            {/* ---------------------------------------------------------- */}
            <div className="section-block">
              <p className="section-title">G-CSF Administration</p>
              <div className="grid-2">
                <FField label="Prophylactic G-CSF Total Dose" unit="μg">
                  <input type="number" name="TOTAL_GLAS_PREDOSE" value={formData.TOTAL_GLAS_PREDOSE} onChange={handleChange} />
                </FField>
              </div>
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Section 4: Chemotherapy drugs (YJ code entry)              */}
            {/* ---------------------------------------------------------- */}
            <div className="section-block">
              <p className="section-title">Chemotherapy Drugs (YJ Code)</p>

              {/* Tags showing currently added drugs with removal buttons */}
              {Object.entries(formData.drugs).length > 0 && (
                <div className="drug-tags">
                  {Object.entries(formData.drugs).map(([code, dose]) => (
                    <span key={code} className="drug-tag">
                      <span className="drug-tag-code">{code}</span>
                      <span className="drug-tag-dose">{dose} mg</span>
                      <button type="button" className="drug-tag-remove" onClick={() => removeDrug(code)}>×</button>
                    </span>
                  ))}
                </div>
              )}

              {/* Drug code search input with autocomplete dropdown */}
              <div className="drug-input-row">
                <div className="drug-search-wrap">
                  <input
                    type="text"
                    className="drug-search-input"
                    placeholder="Enter or search YJ code (e.g. 4291002)"
                    value={drugSearch}
                    onChange={e => { setDrugSearch(e.target.value); setDrugInputCode(e.target.value) }}
                  />
                  {showDropdown && (
                    <ul className="drug-dropdown">
                      {filteredYjCodes.slice(0, 8).map(code => (
                        <li key={code} onClick={() => { setDrugInputCode(code); setDrugSearch(code) }}>{code}</li>
                      ))}
                      {filteredYjCodes.length > 8 && (
                        <li className="drug-dropdown-more">+{filteredYjCodes.length - 8} more…</li>
                      )}
                    </ul>
                  )}
                </div>

                {/* Dose entry field */}
                <input
                  type="number"
                  className="drug-dose-input"
                  placeholder="Dose (mg)"
                  value={drugInputDose}
                  onChange={e => setDrugInputDose(e.target.value)}
                  min="0" step="any"
                  style={{ flex: 1 }}
                />
                <button type="button" className="btn btn-secondary" onClick={addDrug}>Add</button>
              </div>
              <p className="hint-text">Supports 208 YJ-coded chemotherapy and supportive-care agents.</p>
            </div>

            {/* ---------------------------------------------------------- */}
            {/* Section 5: ICD-10 diagnosis code selection                 */}
            {/* ---------------------------------------------------------- */}
            <div className="section-block">
              <p className="section-title">Diagnosis (ICD-10 Chapter C codes)</p>

              {/* Filter input for the diagnosis grid */}
              <input
                type="text"
                className="diag-search-input"
                placeholder="Filter by code or site name (e.g. C34, lung)"
                value={diagSearch}
                onChange={e => setDiagSearch(e.target.value)}
              />

              {/* Summary of currently selected diagnosis codes */}
              {formData.diagnoses.length > 0 && (
                <div className="diag-selected-summary">
                  Selected:&ensp;
                  {formData.diagnoses.map(c => (
                    <span key={c} className="diag-selected-badge" onClick={() => toggleDiag(c)} title="Click to deselect">{c}</span>
                  ))}
                </div>
              )}

              {/* Scrollable diagnosis code grid */}
              <div className="diag-grid">
                {filteredDiagCodes.map(code => (
                  <label key={code} className={`diag-item ${formData.diagnoses.includes(code) ? 'diag-item-selected' : ''}`}>
                    <input type="checkbox" checked={formData.diagnoses.includes(code)} onChange={() => toggleDiag(code)} style={{ display: 'none' }} />
                    <span className="diag-code">{code}</span>
                    <span className="diag-label">{DIAGNOSIS_LABELS[code] || ''}</span>
                  </label>
                ))}
                {filteredDiagCodes.length === 0 && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--gray-500)', gridColumn: '1/-1' }}>No matching codes found.</p>
                )}
              </div>
            </div>

            <button type="submit" className="btn btn-primary submit-btn">Run Prediction</button>
          </form>
        </section>

        {/* ============================================================== */}
        {/* Right panel: Prediction result display                         */}
        {/* ============================================================== */}
        <section className="prediction-result">
          <p className="result-heading">Prediction Result</p>
          {result ? <ResultPanel result={result} /> : <EmptyResult />}
        </section>
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/**
 * FField — Labelled form-field wrapper.
 *
 * Renders a form group consisting of a label (with optional unit annotation)
 * and the provided child element (typically an <input> or <select>).
 *
 * @param {string} label  - Field label text.
 * @param {string} [unit] - Optional measurement unit shown in parentheses.
 * @param {React.ReactNode} children - The form control to render.
 */
function FField({ label, unit, children }) {
  return (
    <div className="form-group">
      <label>
        {label}
        {unit && <span className="unit-text">({unit})</span>}
      </label>
      {children}
    </div>
  )
}

/**
 * ResultPanel — Displays the model prediction output.
 *
 * Shows the FN risk score as a percentage, a colour-coded risk level badge
 * (high / low), and a mandatory clinical disclaimer.
 *
 * @param {{ risk_score: number, level: string }} result - API response object.
 */
function ResultPanel({ result }) {
  const pct    = (result.risk_score * 100).toFixed(1)
  const isHigh = result.level === 'high'

  return (
    <div>
      <div className="risk-score-block">
        {/* Numeric risk score */}
        <div className="risk-score-value" style={{ color: isHigh ? 'var(--red)' : 'var(--green)' }}>
          {pct}<span style={{ fontSize: '1.1rem', fontWeight: 600 }}>%</span>
        </div>
        <p className="risk-score-label-text">FN Risk Score</p>
        {/* Binary risk classification badge */}
        <span className={`risk-level-badge ${isHigh ? 'badge-high' : 'badge-low'}`}>
          {isHigh ? 'High Risk' : 'Low Risk'}
        </span>
      </div>

      {/* Clinical disclaimer — mandatory for AI-assisted decision-support tools */}
      <p className="disclaimer">
        This system is an AI-based clinical decision-support tool. The risk score
        is derived from a LightGBM model trained on 288 patient-level features and
        should be interpreted as supplementary information only. All final clinical
        decisions remain the sole responsibility of the treating physician.
      </p>
    </div>
  )
}

/**
 * EmptyResult — Placeholder shown before the first prediction is submitted.
 */
function EmptyResult() {
  return (
    <div className="empty-result">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 12h6m-3-3v6m-7 4h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2z" />
      </svg>
      <p>Enter patient data in the form on the left and click "Run Prediction".</p>
    </div>
  )
}
