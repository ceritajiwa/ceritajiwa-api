
# -*- coding: utf-8 -*-
"""Cerita Jiwa Platform - Backend API (FastAPI).
Membungkus seluruh logika asesmen/ujian/BEI yang sudah ada sebagai REST API.
Semua secret via environment variables (HF Spaces Secrets)."""
import os, io, json, hmac, hashlib, time, zipfile
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from supabase import create_client

from instruments import INSTRUMENTS, DIMS, DIM_ORDER, INST_SHORT
from scoring import compute_dim_scores, responses_to_answers
from insights import (grouped_insights, gap_analysis, strengths_and_concerns,
                      soft_recommendations, hr_risks, hr_opening, CAPTIONS, closing_paragraph)
from hr_analytics import clusterize, cluster_summary, band_counts_health, build_action_plan, ACTION_META, FLIP, healthify
from questions import EXAM_PACKAGES
from bei import (BEI_PROMPTS, STRUCT_FIELDS, structure_bei, bei_participant_pdf,
                 bei_company_pdf, bei_radar_png)
from charts import radar_chart
from pdf_report import individual_pdf

# ---------- konfigurasi ----------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]
TRAINER_PASSWORD = os.environ.get("TRAINER_PASSWORD") or ADMIN_PASSWORD
SECRET = os.environ.get("API_SECRET", ADMIN_PASSWORD).encode()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Cerita Jiwa Assessment API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

# ---------- auth sederhana (HMAC token, berlaku 12 jam) ----------
def _token(role: str) -> str:
    ts = int(time.time())
    sig = hmac.new(SECRET, f"{role}:{ts}".encode(), hashlib.sha256).hexdigest()
    return f"{role}.{ts}.{sig}"

def _check(token: Optional[str], role: str):
    if not token:
        raise HTTPException(401, "Token diperlukan")
    try:
        r, ts, sig = token.split(".")
        assert r == role and hmac.compare_digest(
            sig, hmac.new(SECRET, f"{r}:{ts}".encode(), hashlib.sha256).hexdigest())
        assert int(time.time()) - int(ts) < 12 * 3600
    except Exception:
        raise HTTPException(401, "Token tidak valid/kedaluwarsa")

class LoginIn(BaseModel):
    password: str
    role: str = "admin"   # admin | trainer

@app.post("/api/login")
def login(body: LoginIn):
    expected = ADMIN_PASSWORD if body.role == "admin" else TRAINER_PASSWORD
    if body.password != expected:
        raise HTTPException(401, "Password salah")
    return {"token": _token(body.role), "role": body.role}

# ---------- katalog & training ----------
@app.get("/api/trainings")
def get_trainings():
    rows = sb.table("trainings").select("*").order("name").execute().data or []
    for t in rows:
        t["enabled"] = _enabled(t)
    return rows

def _enabled(t):
    raw = t.get("enabled_instruments")
    if not raw:
        return [i["key"] for i in INSTRUMENTS]
    try:
        keys = [k for k in json.loads(raw) if any(i["key"] == k for i in INSTRUMENTS)]
        return keys or [i["key"] for i in INSTRUMENTS]
    except Exception:
        return [i["key"] for i in INSTRUMENTS]

@app.get("/api/catalog")
def catalog():
    """Metadata seluruh instrumen untuk dirender frontend."""
    insts = []
    for inst in INSTRUMENTS:
        insts.append(dict(
            key=inst["key"], name=inst["name"], intro=inst["intro"],
            scale=inst.get("scale"), mcq=bool(inst.get("mcq")),
            items=[dict(n=it["n"], text=it.get("text"), opts=it.get("opts"), img=it.get("img"))
                   for it in inst["items"]]))
    dims = {k: dict(label=v["label"], radar=v["radar"], dir=v["dir"],
                    target=v["target"], flip=k in FLIP) for k, v in DIMS.items()}
    return {"instruments": insts, "dims": dims, "dim_order": DIM_ORDER,
            "short": INST_SHORT, "captions": CAPTIONS,
            "action_meta": {k: dict(label=m["label"], color=m["color"], desc=m["desc"])
                            for k, m in ACTION_META.items()},
            "bei_prompts": [dict(k=k, title=t, hint=h) for k, t, h in BEI_PROMPTS],
            "struct_fields": STRUCT_FIELDS}

class TrainingIn(BaseModel):
    name: str
    access_code: Optional[str] = None
    enabled: list = []

@app.post("/api/admin/trainings")
def add_training(body: TrainingIn, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    if not body.name.strip():
        raise HTTPException(400, "Nama wajib")
    row = sb.table("trainings").insert(dict(
        name=body.name.strip(), access_code=body.access_code or None,
        enabled_instruments=json.dumps(body.enabled) if body.enabled else None)).execute().data
    return row

@app.put("/api/admin/trainings/{tid}")
def edit_training(tid: str, body: TrainingIn, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    sb.table("trainings").update(dict(
        name=body.name.strip(), access_code=body.access_code or None,
        enabled_instruments=json.dumps(body.enabled) if body.enabled else None)).eq("id", tid).execute()
    return {"ok": True}

@app.delete("/api/admin/trainings/{tid}")
def del_training(tid: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    sb.table("trainings").delete().eq("id", tid).execute()
    return {"ok": True}

# ---------- asesmen peserta ----------
class CodeIn(BaseModel):
    training_id: str
    code: str

@app.post("/api/verify-code")
def verify_code(body: CodeIn):
    t = (sb.table("trainings").select("*").eq("id", body.training_id).execute().data or [None])[0]
    if not t:
        raise HTTPException(404, "Training tidak ditemukan")
    if t.get("access_code") and body.code != t["access_code"]:
        raise HTTPException(403, "Kode akses salah")
    return {"ok": True}

class SubmitIn(BaseModel):
    training_id: str
    name: str
    email: str
    department: Optional[str] = ""
    job_level: Optional[str] = ""
    access_code: Optional[str] = ""
    answers: dict          # {"PSS:1": 3, ...}  (instrument:item_n -> skor mentah)

@app.post("/api/assessment/submit")
def submit_assessment(body: SubmitIn):
    t = sb.table("trainings").select("*").eq("id", body.training_id).execute().data
    if not t:
        raise HTTPException(404, "Training tidak ditemukan")
    t = t[0]
    if t.get("access_code") and body.access_code != t["access_code"]:
        raise HTTPException(403, "Kode akses salah")
    email = body.email.strip().lower()
    dup = sb.table("respondents").select("id").eq("training_id", body.training_id).eq("email", email).execute().data
    if dup:
        raise HTTPException(409, "Email sudah pernah mengisi untuk training ini")
    ins = sb.table("respondents").insert(dict(
        training_id=body.training_id, full_name=body.name.strip(), email=email,
        department=body.department or None, job_level=body.job_level or None)).execute().data
    rid = ins[0]["id"]
    rows = [dict(respondent_id=rid, instrument=k.split(":")[0], item_n=int(k.split(":")[1]), score=int(v))
            for k, v in body.answers.items()]
    for i in range(0, len(rows), 500):
        sb.table("responses").insert(rows[i:i+500]).execute()
    answers = {(r["instrument"], r["item_n"]): r["score"] for r in rows}
    insts = [i for i in INSTRUMENTS if i["key"] in _enabled(t)]
    scores = compute_dim_scores(answers, instruments=insts)
    return {"respondent_id": rid, "scores": scores, "groups": grouped_insights(scores)}

def _fetch_scores(training_id):
    resps = sb.table("respondents").select("*").eq("training_id", training_id).execute().data or []
    if not resps:
        return None
    ids = [r["id"] for r in resps]
    all_rows, start = [], 0
    for i in range(0, len(ids), 200):
        part = ids[i:i+200]; offset = 0
        while True:
            res = sb.table("responses").select("*").in_("respondent_id", part).order("id").range(offset, offset+999).execute()
            rows = res.data or []
            all_rows.extend(rows)
            if len(rows) < 1000:
                break
            offset += 1000
    # bangun df skor
    import pandas as pd
    from scoring import group_scores as _gs
    df = _gs(pd.DataFrame(all_rows))
    if df.empty:
        return None
    df = df.merge(pd.DataFrame(resps)[["id", "full_name", "department", "job_level"]],
                  left_on="respondent_id", right_on="id", how="left")
    return df, resps

@app.get("/api/report/company/{training_id}")
def company_report(training_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    out = _fetch_scores(training_id)
    if not out:
        return {"empty": True}
    df, resps = out
    import pandas as pd
    cols = [d for d in DIM_ORDER if d in df.columns]
    mean = {d: round(float(df[d].mean()), 1) for d in cols}
    gap_rows, priority = gap_analysis(mean)
    strengths, concerns = strengths_and_concerns(gap_rows)
    idx = clusterize(df)
    counts = idx["cluster"].value_counts().to_dict()
    df_h = df.copy()
    for d in cols:
        if d in FLIP:
            df_h[d] = 100 - df_h[d]
    actions = build_action_plan(gap_rows)
    return {
        "n": len(df), "dims": cols, "mean": mean,
        "gap": gap_rows, "strengths": strengths, "concerns": concerns,
        "recommendations": soft_recommendations(priority),
        "risks": hr_risks(concerns),
        "clusters": {"counts": counts, "summary": cluster_summary(idx),
                     "members": {int(c): idx[idx["cluster"] == c][["full_name", "department", "Indeks Keseluruhan", "area_rawan"]].to_dict("records")
                                 for c in counts}},
        "band": band_counts_health(df_h, cols),
        "actions": {k: [dict(label=r["label"], gap=r["gap"], rekomendasi=r["rekomendasi"], dim=r["dim"]) for r in v]
                    for k, v in actions.items()},
    }

# PDF report perusahaan (reuse generator Streamlit -> versi ringkas dulu: kirim data JSON saja,
# frontend render grafik; PDF lengkap menyusul di tahap berikutnya)

# ---------- ujian ----------
@app.get("/api/exam/packages")
def exam_packages():
    return {k: dict(title=v["title"], passing=v["passing"], sessions=v["sessions"],
                    questions=[dict(n=q["n"], session=q["session"], q=q["q"], opts=q["opts"])
                               for q in v["questions"]])   # key tidak dikirim!
                   for k, v in EXAM_PACKAGES.items()}

class ExamIn(BaseModel):
    training_id: str
    package: str
    name: str
    email: str
    access_code: Optional[str] = ""
    answers: dict      # {no: index 0-3}

@app.post("/api/exam/submit")
def exam_submit(body: ExamIn):
    pkg = EXAM_PACKAGES.get(body.package)
    if not pkg:
        raise HTTPException(404, "Paket ujian tidak ditemukan")
    t = (sb.table("trainings").select("*").eq("id", body.training_id).execute().data or [None])[0]
    if t and t.get("access_code") and body.access_code != t["access_code"]:
        raise HTTPException(403, "Kode akses salah")
    correct = sum(1 for q in pkg["questions"] if body.answers.get(str(q["n"])) == q["key"])
    score = round(correct / len(pkg["questions"]) * 100)
    passed = score >= pkg["passing"]
    sb.table("exam_attempts").insert(dict(
        training_id=body.training_id, full_name=body.name.strip(),
        email=body.email.strip().lower(), correct=correct, score=score,
        passed=passed, package=body.package)).execute()
    return {"correct": correct, "score": score, "passed": passed}

@app.get("/api/admin/exam/{training_id}")
def exam_history(training_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    return sb.table("exam_attempts").select("*").eq("training_id", training_id).order("created_at", desc=True).execute().data or []

# ---------- BEI ----------
@app.get("/api/bei/{training_id}")
def bei_list(training_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "trainer")
    return sb.table("bei_sessions").select("*").eq("training_id", training_id).order("created_at", desc=True).execute().data or []

class BeiIn(BaseModel):
    training_id: str
    participant_name: str
    counselor_name: str
    narratives: dict
    session_id: Optional[str] = None   # isi = update, kosong = baru

@app.post("/api/bei/save")
def bei_save(body: BeiIn, authorization: Optional[str] = Header(None)):
    _check(authorization, "trainer")
    structured = structure_bei(body.narratives, GEMINI_KEY, GROQ_KEY)
    if body.session_id:
        sb.table("bei_sessions").update(dict(narratives=body.narratives, structured=structured)).eq("id", body.session_id).execute()
    else:
        sb.table("bei_sessions").insert(dict(
            training_id=body.training_id, participant_name=body.participant_name.strip(),
            counselor_name=body.counselor_name.strip(), narratives=body.narratives,
            structured=structured)).execute()
    return {"structured": structured}

@app.get("/api/bei/pdf/{session_id}")
def bei_pdf(session_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "trainer")
    s = sb.table("bei_sessions").select("*").eq("id", session_id).execute().data
    if not s:
        raise HTTPException(404, "Sesi tidak ditemukan")
    s = s[0]
    pdf = bei_participant_pdf("Cerita Jiwa", s["participant_name"], s.get("counselor_name", "-"),
                              s.get("narratives") or {}, s.get("structured"),
                              tanggal=str(s.get("created_at", "")))
    return Response(content=pdf if isinstance(pdf, bytes) else pdf.encode("latin-1"),
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=BEI_{s['participant_name']}.pdf"})


@app.get("/api/assessment/{rid}/pdf")
def assessment_pdf(rid: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    resp = (sb.table("respondents").select("*").eq("id", rid).execute().data or [None])[0]
    if not resp:
        raise HTTPException(404, "Responden tidak ditemukan")
    rows = sb.table("responses").select("*").eq("respondent_id", rid).execute().data or []
    t = (sb.table("trainings").select("*").eq("id", resp["training_id"]).execute().data or [{}])[0]
    answers = {(r["instrument"], r["item_n"]): r["score"] for r in rows}
    insts = [i for i in INSTRUMENTS if i["key"] in _enabled(t)]
    scores = compute_dim_scores(answers, instruments=insts)
    groups = grouped_insights(scores)
    png = radar_chart(scores, f"Profil {resp['full_name']}")
    pdf = individual_pdf(resp["full_name"], t.get("name", ""), resp.get("department"),
                         resp.get("job_level"), scores, groups, png)
    data = pdf if isinstance(pdf, bytes) else pdf.encode("latin-1")
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f"attachment; filename=Hasil_{resp['full_name'].replace(' ', '_')}.pdf"})

@app.get("/api/admin/respondents/{training_id}")
def list_respondents(training_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    return sb.table("respondents").select("*").eq("training_id", training_id).order("created_at").execute().data or []


# ---------- email & sertifikat ----------
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from fpdf import FPDF

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_FROM = os.environ.get("MAIL_FROM") or SMTP_USER

class CertPDF(FPDF):
    def footer(self):
        self.set_y(-16)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "Certified by Cerita Jiwa Training Center", align="C")

def _certificate_pdf(name: str, program: str) -> bytes:
    pdf = CertPDF("L")
    pdf.add_page("L")
    pdf.set_fill_color(52, 74, 97)
    pdf.rect(0, 0, 297, 70, "F")
    try:
        pdf.image("logo-white.png", x=128, y=12, w=42)
    except Exception:
        pass
    pdf.set_y(78)
    pdf.set_font("helvetica", "B", 30)
    pdf.set_text_color(52, 74, 97)
    pdf.cell(0, 14, "Certificate of Completion", align="C", ln=1)
    pdf.ln(4)
    pdf.set_font("helvetica", "", 13)
    pdf.set_text_color(109, 111, 113)
    pdf.cell(0, 8, "This certificate is proudly presented to", align="C", ln=1)
    pdf.ln(3)
    pdf.set_font("helvetica", "B", 26)
    pdf.set_text_color(52, 74, 97)
    pdf.cell(0, 14, _pdf_safe(name), align="C", ln=1)
    pdf.ln(3)
    pdf.set_font("helvetica", "", 13)
    pdf.set_text_color(109, 111, 113)
    pdf.cell(0, 8, "for successfully completing", align="C", ln=1)
    pdf.set_font("helvetica", "B", 17)
    pdf.set_text_color(52, 74, 97)
    pdf.cell(0, 11, _pdf_safe(program), align="C", ln=1)
    pdf.ln(6)
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(109, 111, 113)
    pdf.cell(0, 8, datetime.now().strftime("%d %B %Y"), align="C", ln=1)
    pdf.ln(10)
    pdf.set_draw_color(111, 194, 180)
    pdf.set_line_width(0.8)
    pdf.line(118, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 7, "Cerita Jiwa Training Center", align="C", ln=1)
    out = pdf.output()
    return out if isinstance(out, bytes) else out.encode("latin-1")

def _pdf_safe(s):
    return (str(s).replace("\\u2019", "'").replace("\\u2018", "'")
            .replace("\\u201c", '"').replace("\\u201d", '"')
            .encode("latin-1", errors="replace").decode("latin-1"))

def _send_mail(to: str, subject: str, body: str, attachments: list):
    """attachments: list of (filename, bytes). Raise kalau SMTP belum dikonfigurasi."""
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise HTTPException(501, "Email belum dikonfigurasi: isi SMTP_HOST/SMTP_USER/SMTP_PASS di Environment Variables Render.")
    msg = MIMEMultipart()
    msg["From"] = MAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for fn, data in attachments:
        part = MIMEApplication(data, _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=fn)
        msg.attach(part)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(MAIL_FROM, [to], msg.as_string())

@app.post("/api/admin/email/assessment/{rid}")
def email_assessment(rid: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    pdf_data = assessment_pdf(rid, authorization).body
    if not isinstance(pdf_data, bytes):
        pdf_data = bytes(pdf_data)
    resp = (sb.table("respondents").select("*").eq("id", rid).execute().data or [None])[0]
    if not resp:
        raise HTTPException(404, "Responden tidak ditemukan")
    _send_mail(resp["email"],
               f"Hasil Asesmen Anda - Cerita Jiwa",
               f"Halo {resp['full_name']},\\n\\nTerima kasih telah mengikuti asesmen. Berikut lampiran laporan hasil asesmen Anda.\\n\\nSalam hangat,\\nCerita Jiwa Training Center",
               [(f"Hasil_Asesmen_{resp['full_name'].replace(' ', '_')}.pdf", pdf_data)])
    return {"ok": True, "to": resp["email"]}

@app.post("/api/admin/email/certificate/{attempt_id}")
def email_certificate(attempt_id: str, authorization: Optional[str] = Header(None)):
    _check(authorization, "admin")
    a = (sb.table("exam_attempts").select("*").eq("id", attempt_id).execute().data or [None])[0]
    if not a:
        raise HTTPException(404, "Percobaan ujian tidak ditemukan")
    if not a.get("passed"):
        raise HTTPException(400, "Peserta belum lulus - sertifikat hanya untuk yang lulus.")
    program = EXAM_PACKAGES.get(a.get("package"), {}).get("title", "Training Cerita Jiwa")
    cert = _certificate_pdf(a["full_name"], program)
    _send_mail(a["email"],
               f"Sertifikat Kelulusan Anda - Cerita Jiwa",
               f"Selamat {a['full_name']}!\\n\\nAnda dinyatakan LULUS ({program}) dengan nilai {a['score']}.\\nLampiran: sertifikat PDF + badge yang dapat Anda cantumkan di LinkedIn (bagian Licenses & Certifications).\\n\\nSalam hangat,\\nCerita Jiwa Training Center",
               [(f"Sertifikat_{a['full_name'].replace(' ', '_')}.pdf", cert)])
    return {"ok": True, "to": a["email"]}

@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}
