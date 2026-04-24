import os
from flask import (
    Flask, render_template, request, redirect, session, flash,
    send_file, url_for, jsonify, send_from_directory
)
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer

# local ml helper
import ml_model

# ---------- CONFIG ----------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey123")

COMMENTS_FILE = "comments.csv"
MODEL_FILE = "model.pkl"

TWO_FACTOR_API_KEY = os.getenv("TWO_FACTOR_API_KEY", "9827fbd6-d5c4-11f0-a6b2-0200cd936042")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin")

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ---------- DRAFTS DATA ----------
DRAFTS = [
    {
        "id": "dpb",
        "title": "Data Protection Bill",
        "description": "Rules about data privacy & consent.",
        "image": "download.jpeg",
        "full_text": (
            "The Data Protection Bill focuses on protecting citizens' personal data, "
            "ensuring privacy, consent rights, and stringent data governance norms."
        ),
        "pdf": "data_protection_bill.pdf",
        "points": [
            "Consent required for data collection",
            "Right to erase personal data",
            "Mandatory breach notification",
            "Cross-border data rules"
        ],
        "flow": ["Data Collection → Secure Storage → Consent Check → User Rights"]
    },
    {
        "id": "cga",
        "title": "Corporate Governance Act",
        "description": "Board governance structure changes.",
        "image": "download (1).jpeg",
        "full_text": "Improves transparency and mandates audit controls for companies.",
        "pdf": "corporate_governance_act.pdf",
        "points": [
            "Independent directors required",
            "Audit committee updates",
            "Conflict of interest disclosure",
            "Annual board evaluation"
        ],
        "flow": ["Board Formation → Audit Committee → Compliance → Disclosure"]
    },
    {
        "id": "ca2025",
        "title": "Companies Amendment 2025",
        "description": "Major updates in compliance.",
        "image": "download (2).jpeg",
        "full_text": "Updates compliance process, penalties, and CSR reporting.",
        "pdf": "companies_amendment_2025.pdf",
        "points": [
            "Quick incorporation",
            "Reduced penalties",
            "Online compliance submissions",
            "Updated CSR rules"
        ],
        "flow": ["Incorporation → Filing → CSR Reporting → Audit"]
    },
    {
        "id": "dmf",
        "title": "Digital Markets Framework",
        "description": "Regulation for digital platforms.",
        "image": "images.jpeg",
        "full_text": "Ensures fair competition and transparency for digital platforms.",
        "pdf": "digital_markets_framework.pdf",
        "points": [
            "Data transparency",
            "Fair advertising rules",
            "Algorithm accountability",
            "Anti-monopoly measures"
        ],
        "flow": ["Platform → Data Analysis → Fair Use → Monitoring"]
    },
    {
        "id": "sis",
        "title": "Startup Incentive Scheme",
        "description": "Incentives for new startups.",
        "image": "download.png",
        "full_text": "Provides tax benefits, funding support, and incubation services.",
        "pdf": "startup_incentive_scheme.pdf",
        "points": [
            "3-year tax exemption",
            "Incubation support",
            "Fast trademark approval",
            "Seed funding"
        ],
        "flow": ["Register → Funding → Incubation → Growth"]
    },
    {
        "id": "csr",
        "title": "CSR Amendment",
        "description": "CSR contribution rules updated.",
        "image": "download (1).png",
        "full_text": "Mandates CSR audit, reporting transparency, and sustainability focus.",
        "pdf": "csr_amendment.pdf",
        "points": [
            "Mandatory CSR audit",
            "Unused funds must be reported",
            "Eligibility expansion",
            "Compliance tracking"
        ],
        "flow": ["Budgeting → Allocation → Execution → Audit"]
    }
]


# CSV schema used by app (keeps backward compatibility)
CSV_COLUMNS = [
    "Draft", "Mobile", "Comment", "EntityType", "Name", "Email", "Company"
]

# ---------- HELPERS ----------
USERS_FILE = "users.csv"

def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=["Name","Mobile","Email"]).to_csv(USERS_FILE, index=False)

def ensure_comments_file():
    """Create comments file with the full columns if missing."""
    if not os.path.exists(COMMENTS_FILE):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(COMMENTS_FILE, index=False)
    else:
        # If file exists but lacks some columns, try to add missing columns preserving data
        df = pd.read_csv(COMMENTS_FILE)
        missing = [c for c in CSV_COLUMNS if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = ""
            df.to_csv(COMMENTS_FILE, index=False)

def save_comment_row(draft, mobile, comment,
                     entity_type="individual", name="", email="", company=""):
    """Append a row to the CSV with the full schema."""
    ensure_comments_file()
    row = {
        "Draft": draft,
        "Mobile": mobile,
        "Comment": comment,
        "EntityType": entity_type,
        "Name": name,
        "Email": email,
        "Company": company
    }
    pd.DataFrame([row]).to_csv(COMMENTS_FILE, mode="a", header=False, index=False)

def find_draft_by_id(value):
    if not value:
        return None
    value_norm = str(value).replace("%20", " ").strip().lower()
    for d in DRAFTS:
        if d["id"].lower() == value_norm or d["title"].lower() == value_norm:
            return d
    # fallback – match by partial title
    for d in DRAFTS:
        if value_norm in d["title"].lower():
            return d
    return None

def load_saved_model():
    """Try loading a saved joblib model; return None on failure."""
    if os.path.exists(MODEL_FILE):
        try:
            saved = joblib.load(MODEL_FILE)
            # saved expected to be dict with keys: model, vectorizer (older saves may differ)
            if isinstance(saved, dict) and "model" in saved:
                return saved
            # if a single pipeline object was saved, return a compatible dict
            return {"model": saved, "vectorizer": None}
        except Exception:
            return None
    return None

# ---------- ROUTES ----------

@app.before_request
def clear_session_on_restart():
    if not session.get("initialized"):
        session.clear()
        session["initialized"] = True

@app.route("/")
def home():
    return render_template("drafts.html", drafts=DRAFTS)

@app.route("/start-comment/<draft_id>")
def start_comment(draft_id):
    if not session.get("user"):
        flash("Login required", "danger")
        return redirect("/login")

    d = find_draft_by_id(draft_id)

    if not d:
        flash("Draft not found", "danger")
        return redirect("/")

    return redirect(url_for("submit_comment", draft=d["title"]))

@app.route("/view-more")
def view_more():
    draft_title = request.args.get("draft", "")
    selected = next((d for d in DRAFTS if d["title"].lower() == draft_title.lower()), None)
    return render_template("view_more.html", draft=draft_title, data=selected)

@app.route("/verify")
def verify_user():
    draft = request.args.get("draft", "Unknown")
    return render_template("verify.html", draft=draft)

# ---------- OTP (2Factor) ----------
@app.route("/send-otp", methods=["POST"])
def send_otp():
    mobile = request.form.get("mobile")
    if not mobile:
        return jsonify({"status": "error", "message": "Mobile required"}), 400

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/{mobile}/AUTOGEN/OTP1"
    try:
        res = requests.get(url, timeout=8).json()
    except Exception:
        return jsonify({"status": "error", "message": "OTP service error"}), 500

    if res.get("Status") == "Success":
        session["OTP_SESSION_ID"] = res.get("Details")
        return jsonify({"status": "success", "message": "OTP sent via SMS"})
    return jsonify({"status": "error", "message": res.get("Details", "Failed to send")}), 400

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    otp = request.form.get("otp")
    mobile = request.form.get("mobile")
    draft = request.form.get("draft", "")
    sid = session.get("OTP_SESSION_ID")
    if not sid:
        return jsonify({"status": "error", "message": "Session expired. Send OTP again."}), 400

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/VERIFY/{sid}/{otp}"
    try:
        res = requests.get(url, timeout=8).json()
    except Exception:
        return jsonify({"status": "error", "message": "Verification service error"}), 500

    if res.get("Status") == "Success":
        redirect_url = f"/submit-comment?draft={draft}&mobile={mobile}"
        return jsonify({"status": "success", "redirect": redirect_url})
    return jsonify({"status": "error", "message": "Invalid OTP"}), 400

# ---------- COMMENT FORM ----------
@app.route("/submit-comment", methods=["GET", "POST"])
def submit_comment():
    if request.method == "GET":
        draft = request.args.get("draft", "")
        mobile = request.args.get("mobile", "")
        draft_obj = find_draft_by_id(draft)
        about = draft_obj["description"] if draft_obj else ""
        return render_template("submit_comment.html", draft=draft, mobile=mobile, about=about)

    # POST -> collect fields and save
    draft = request.form.get("draft", "")
    mobile = request.form.get("mobile", "")
    comment = request.form.get("comment", "").strip()
    entity_type = request.form.get("entityType", "individual")  # individual or company
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    company = request.form.get("company", "").strip()

    if not draft or not comment:
        flash("Draft and comment are required", "danger")
        return redirect(request.url)

    save_comment_row(draft, mobile, comment,
                     entity_type=entity_type, name=name, email=email, company=company)
    flash("Comment submitted — thank you!", "success")
    return render_template("submit_success.html")

# ---------- ADMIN ----------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
            session["admin"] = True
            return redirect("/dashboard")
        flash("Invalid credentials", "danger")
    return render_template("admin_login.html")

@app.route("/admin-logout")
def admin_logout():
    session.pop("admin", None)
    flash("Logged out successfully", "success")
    return redirect(url_for("home"))   # drafts page

@app.route("/toggle-theme")
def toggle_theme():
    if not session.get("admin"):
        return redirect("/admin")
    session["theme"] = "dark" if session.get("theme", "light") == "light" else "light"
    return redirect("/dashboard")

# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    if not session.get("admin"):
        return redirect("/admin")
    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)

    # optional filter by draft
    selected = request.args.get("draft", "")
    if selected:
        df = df[df["Draft"].str.lower() == selected.lower()]

    # normalize column names if needed
    if "Comment" not in df.columns:
        for alt in ["comment", "Comments", "Text"]:
            if alt in df.columns:
                df.rename(columns={alt: "Comment"}, inplace=True)
                break

    # Use VADER for base scoring
    analyzer = SentimentIntensityAnalyzer()
    if not df.empty:
        df["clean"] = df["Comment"].astype(str).str.replace(r"\s+", " ", regex=True)
    else:
        df["clean"] = []

    df["score"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"] if x else 0.0)

    def lab(s):
        if s >= 0.05: return "Positive"
        if s <= -0.05: return "Negative"
        return "Neutral"

    df["Sentiment"] = df["score"].apply(lab)

    totals = {
        "total": int(len(df)),
        "pos": int((df["Sentiment"] == "Positive").sum()),
        "neg": int((df["Sentiment"] == "Negative").sum()),
        "neu": int((df["Sentiment"] == "Neutral").sum())
    }

    # wordcloud generation (saved image path)
    text = " ".join(df["clean"].astype(str).tolist())
    wordcloud_url = None
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color="white", stopwords=set(STOPWORDS)).generate(text)
        wc.to_file("static/wordcloud.png")
        wordcloud_url = url_for("static", filename="wordcloud.png")

    # render admin dashboard (pass drafts list so admin can select)
    return render_template(
        "admin_dashboard.html",
        drafts=DRAFTS,
        selected=selected,
        totals=totals,
        comments=df.to_dict(orient="records"),
        wordcloud_url=wordcloud_url,
        theme=session.get("theme", "light")
    )

# ---------- TRAIN (manual) ----------
@app.route("/train", methods=["POST"])
def train():
    if not session.get("admin"):
        return redirect("/admin")
    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)
    if df.empty:
        flash("No comments to train", "danger")
        return redirect("/dashboard")

    # use VADER labels as weak labels to train a quick ML model (as before)
    analyzer = SentimentIntensityAnalyzer()
    df["clean"] = df["Comment"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
    df["score"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    def label_map(s):
        if s >= 0.05: return "positive"
        if s <= -0.05: return "negative"
        return "neutral"
    df["label"] = df["score"].apply(label_map)

    # train small model using ml_model helpers
    X = df["clean"].tolist()
    y = df["label"].tolist()

    # use ml_model.train_model_from_data to produce a pipeline
    model_pipeline = ml_model.train_model_from_data(X, y)

    # save model
    joblib.dump({"model": model_pipeline, "vectorizer": None}, MODEL_FILE)
    flash("Training completed and model saved.", "success")
    return redirect("/dashboard")

# ---------- ANALYZE (charts / full comment list) ----------
@app.route("/analyze")
def analyze():
    if not session.get("admin"):
        return redirect("/admin")
    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)

    # optional filter by draft chosen on dashboard
    selected = request.args.get("draft", "")
    if selected:
        df = df[df["Draft"].str.lower() == selected.lower()]

    # normalize comment column
    if "Comment" not in df.columns:
        for alt in ["comment", "Comments", "Text"]:
            if alt in df.columns:
                df.rename(columns={alt: "Comment"}, inplace=True)
                break

    analyzer = SentimentIntensityAnalyzer()
    df["clean"] = df["Comment"].astype(str)
    df["score"] = df["clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    def lab(s):
        if s >= 0.05: return "Positive"
        if s <= -0.05: return "Negative"
        return "Neutral"
    df["Sentiment"] = df["score"].apply(lab)

    # counts for charts
    pos = int((df["Sentiment"] == "Positive").sum())
    neg = int((df["Sentiment"] == "Negative").sum())
    neu = int((df["Sentiment"] == "Neutral").sum())

    # Try to augment with ML model predictions (if present)
    saved = load_saved_model()
    ml_preds = None
    if saved:
        try:
            model = saved["model"]
            # if pipeline has vectorizer/estimator inside, it will accept raw text
            ml_preds = model.predict(df["clean"].astype(str).tolist()).tolist()
        except Exception:
            ml_preds = None

    return render_template("analyze.html", pos=pos, neg=neg, neu=neu, comments=df.to_dict(orient="records"), ml_preds=ml_preds)

# ---------- WORDCLOUD PAGE ----------
@app.route("/wordcloud")
def wordcloud_page():
    if not session.get("admin"):
        return redirect("/admin")

    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)

    if df.empty:
        return render_template("wordcloud.html", image_url=None)

    text = " ".join(df["Comment"].astype(str))

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200
    ).generate(text)

    wc.to_file("static/wordcloud.png")

    return render_template("wordcloud.html",
                           image_url=url_for("static", filename="wordcloud.png"))


# ---------- SUMMARY PAGE ----------
@app.route("/summary")
def summary():
    if not session.get("admin"):
        return redirect("/admin")
    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)

    # per comment short summary (take up to 30 words)
    def short_summary(text, nwords=30):
        s = str(text).strip()
        parts = s.split()
        if len(parts) <= nwords:
            return s
        return " ".join(parts[:nwords]) + "..."

    df["short"] = df["Comment"].apply(lambda t: short_summary(t, 30))

    # overall summary: top 8 keywords by frequency (very simple)
    text = " ".join(df["Comment"].astype(str).tolist()).lower()
    stop = set(STOPWORDS)
    words = [w.strip(".,!?:;()[]\"'") for w in text.split() if w.isalpha() and w not in stop]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
    overall_summary = ", ".join([w for w,_ in top_words])

    return render_template("summary.html", comments=df.to_dict(orient="records"), overall=overall_summary)

@app.route("/download-summary")
def download_summary():
    if not session.get("admin"):
        return redirect("/admin")

    ensure_comments_file()
    df = pd.read_csv(COMMENTS_FILE)

    if df.empty:
        flash("No comments found", "danger")
        return redirect("/summary")

    text = "\n\n".join(df["Comment"].astype(str).tolist())

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return send_file("summary.txt", as_attachment=True)

# ---------- DOWNLOAD COMMENTS ----------
@app.route("/download-comments")
def download_comments():
    if not session.get("admin"):
        return redirect("/admin")
    ensure_comments_file()
    return send_file(COMMENTS_FILE, as_attachment=True)

# ---------- PREDICT API ----------
@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("admin"):
        return jsonify({"error": "unauthorized"}), 403
    saved = load_saved_model()
    if not saved:
        return jsonify({"error": "no model"}), 400
    model = saved["model"]
    text = request.form.get("text", "")
    pred = model.predict([text])[0]
    return jsonify({"prediction": pred})

# ---------- SERVE PDFs ----------
@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory(os.path.join(app.static_folder, "pdfs"), filename)

@app.route("/register")
def register_page():
    return render_template("user_register.html")

@app.route("/register-user", methods=["POST"])
def register_user():
    name = request.form.get("name")
    mobile = request.form.get("mobile")
    email = request.form.get("email")

    ensure_users_file()
    df = pd.read_csv(USERS_FILE)

    # check duplicate
    if mobile in df["Mobile"].astype(str).values:
        return jsonify({"status":"exists","message":"User already exists"})

    # save temp user (NOT permanent)
    session["temp_user"] = {
        "Name": name,
        "Mobile": mobile,
        "Email": email
    }

    # 🔥 SEND OTP
    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/{mobile}/AUTOGEN/OTP1"
    res = requests.get(url).json()

    if res.get("Status") == "Success":
        session["OTP_SESSION_ID"] = res.get("Details")
        return jsonify({"status":"otp_sent"})

    return jsonify({"status":"error","message":"OTP failed"})

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("user_login.html")

@app.route("/verify-register-otp", methods=["POST"])
def verify_register_otp():
    otp = request.form.get("otp")
    sid = session.get("OTP_SESSION_ID")

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/VERIFY/{sid}/{otp}"
    res = requests.get(url).json()

    if res.get("Status") == "Success":
        user = session.get("temp_user")

        df = pd.read_csv(USERS_FILE)
        new = pd.DataFrame([user])
        new.to_csv(USERS_FILE, mode="a", header=False, index=False)

        session["user"] = user["Mobile"]
        session.pop("temp_user", None)

        return jsonify({"status":"success","redirect":"/"})

    return jsonify({"status":"error","message":"Invalid OTP"})
@app.route("/send-login-otp", methods=["POST"])
def send_login_otp():
    mobile = request.form.get("mobile")

    df = pd.read_csv(USERS_FILE)

    if mobile not in df["Mobile"].astype(str).values:
        return jsonify({"status":"error","message":"User not registered"})

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/{mobile}/AUTOGEN/OTP1"
    res = requests.get(url).json()

    if res.get("Status") == "Success":
        session["OTP_SESSION_ID"] = res.get("Details")
        session["login_mobile"] = mobile
        return jsonify({"status":"success"})

    return jsonify({"status":"error"})

@app.route("/resend-otp", methods=["POST"])
def resend_otp():
    mobile = None

    # 🔥 check from temp user (registration case)
    if session.get("temp_user"):
        mobile = session["temp_user"]["Mobile"]

    # fallback (login case future use)
    elif session.get("login_mobile"):
        mobile = session.get("login_mobile")

    if not mobile:
        return jsonify({"status":"error","message":"Session expired"}), 400

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/{mobile}/AUTOGEN/OTP1"

    try:
        res = requests.get(url).json()
    except:
        return jsonify({"status":"error","message":"OTP service error"}), 500

    if res.get("Status") == "Success":
        session["OTP_SESSION_ID"] = res.get("Details")
        return jsonify({"status":"success","message":"OTP resent successfully"})

    return jsonify({"status":"error","message":"Failed to resend OTP"})

@app.route("/login-verify", methods=["POST"])
def login_verify():
    otp = request.form.get("otp")
    sid = session.get("OTP_SESSION_ID")

    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/VERIFY/{sid}/{otp}"
    res = requests.get(url).json()

    if res.get("Status") == "Success":
        session["user"] = session.get("login_mobile")
        return jsonify({"status":"success","redirect":"/"})

    return jsonify({"status":"error","message":"Invalid OTP"})

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "success")
    return redirect("/")

@app.route("/profile")
def user_profile():
    if not session.get("user"):
        return redirect("/login")

    mobile = session.get("user")

    df = pd.read_csv(USERS_FILE)
    comments_df = pd.read_csv(COMMENTS_FILE)

    user = df[df["Mobile"].astype(str) == mobile]

    if user.empty:
        flash("User not found","danger")
        return redirect("/login")

    name = user["Name"].values[0]

    user_comments = comments_df[comments_df["Mobile"].astype(str) == mobile]

    return render_template("profile.html",
                           name=name,
                           mobile=mobile,
                           comments=user_comments.to_dict(orient="records"))

# ---------- RUN ----------


if __name__ == "__main__":
    ensure_comments_file()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
