import argparse
import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import torch
import torch.nn as nn


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    if name == "torchvision" or str(name).startswith("torchvision."):
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    from transformers import AutoModel, AutoTokenizer, logging as transformers_logging
finally:
    importlib.util.find_spec = _original_find_spec

transformers_logging.set_verbosity_error()

try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None

if disable_progress_bars is not None:
    disable_progress_bars()


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "roberta_da_course"
CLASSIFIER_HEAD_PATH = BASE_DIR / "model" / "classifier_head.pt"
SUMMARY_PATH = BASE_DIR / "model" / "summary.json"
SUBMISSIONS_PATH = BASE_DIR / "submissions" / "course_survey_submissions.jsonl"
MODEL_NAME = "roberta_da_course_text_plus_scores_joint"

with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
    SUMMARY = json.load(handle)

SCORE_COLUMNS = list(SUMMARY["score_columns"])
SCORE_MEANS = dict(SUMMARY["score_normalization"]["means"])
SCORE_STDS = dict(SUMMARY["score_normalization"]["stds"])
WARNING_THRESHOLD = 0.5
BLOCK_THRESHOLD = 0.8


def render_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Course Feedback Survey</title>
  <style>
    :root {
      --bg: #edf3fb;
      --surface: #ffffff;
      --surface-soft: #f6f9fd;
      --ink: #102345;
      --muted: #65748d;
      --line: #d6e0ef;
      --brand: #0d6fd4;
      --brand-dark: #0a4485;
      --warning: #b46d00;
      --danger: #c22a3d;
      --success: #177457;
      --shadow: 0 24px 70px rgba(16, 35, 69, 0.12);
      --radius-xl: 24px;
      --radius-lg: 18px;
      --radius-md: 14px;
      --font: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --max: 940px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: var(--font);
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(13, 111, 212, 0.12), transparent 28%),
        linear-gradient(180deg, #f5f8fd 0%, #edf3fb 48%, #e8eef8 100%);
    }

    .page {
      min-height: 100vh;
      padding: 28px 18px 36px;
    }

    .survey-shell {
      width: min(var(--max), 100%);
      margin: 0 auto;
    }

    .survey-header {
      margin-bottom: 18px;
      text-align: center;
    }

    .survey-header h1 {
      margin: 0;
      font-size: clamp(2rem, 4.8vw, 3rem);
      font-weight: 300;
      letter-spacing: -0.03em;
    }

    .survey-header p {
      margin: 10px auto 0;
      max-width: 56ch;
      color: var(--muted);
      line-height: 1.65;
      font-size: 0.98rem;
    }

    .survey-card {
      background: rgba(255, 255, 255, 0.94);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      padding: 28px;
      backdrop-filter: blur(10px);
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 18px;
    }

    .topbar h2 {
      margin: 0;
      font-size: 1.45rem;
      font-weight: 600;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      background: #eaf3ff;
      color: var(--brand-dark);
      font-weight: 700;
      font-size: 0.92rem;
      white-space: nowrap;
    }

    .progress {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }

    .progress-step {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #fbfdff;
    }

    .progress-step.active {
      border-color: var(--brand);
      background: linear-gradient(180deg, #eef6ff, #ffffff);
      box-shadow: 0 10px 24px rgba(13, 111, 212, 0.08);
    }

    .progress-step small {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .progress-step strong {
      display: block;
      font-size: 0.98rem;
    }

    .banner {
      display: none;
      margin-bottom: 16px;
      padding: 14px 16px;
      border-radius: 16px;
      font-weight: 600;
      line-height: 1.55;
    }

    .banner.show { display: block; }
    .banner.error {
      background: rgba(194, 42, 61, 0.08);
      color: var(--danger);
      border: 1px solid rgba(194, 42, 61, 0.15);
    }
    .banner.warning {
      background: rgba(180, 109, 0, 0.08);
      color: var(--warning);
      border: 1px solid rgba(180, 109, 0, 0.16);
    }
    .banner.success {
      background: rgba(23, 116, 87, 0.08);
      color: var(--success);
      border: 1px solid rgba(23, 116, 87, 0.15);
    }

    .step-panel { display: none; }
    .step-panel.active { display: block; }

    .section + .section {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
    }

    .section h3 {
      margin: 0 0 6px;
      font-size: 1.08rem;
      font-weight: 600;
    }

    .section-note {
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.94rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }

    .field, .field-full {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .field-full { grid-column: 1 / -1; }

    label {
      font-size: 0.95rem;
      font-weight: 600;
    }

    .hint {
      margin: -2px 0 0;
      color: var(--muted);
      font-size: 0.89rem;
      line-height: 1.55;
    }

    input, select, textarea {
      width: 100%;
      padding: 14px 15px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font: inherit;
      transition: border-color 160ms ease, box-shadow 160ms ease;
    }

    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--brand);
      box-shadow: 0 0 0 4px rgba(13, 111, 212, 0.15);
    }

    textarea {
      min-height: 180px;
      resize: vertical;
      line-height: 1.65;
    }

    .rating-group {
      display: grid;
      gap: 10px;
    }

    .rating-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .rating-chip {
      min-width: 52px;
      padding: 12px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--brand-dark);
      font-weight: 700;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease, color 160ms ease;
    }

    .rating-chip:hover {
      transform: scale(1.05);
      border-color: var(--brand);
    }

    .rating-chip.active {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
      box-shadow: 0 12px 26px rgba(13, 111, 212, 0.2);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .summary-tile {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--surface-soft);
    }

    .summary-tile strong {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .actions {
      margin-top: 26px;
      padding-top: 20px;
      border-top: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    .actions-group {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    button {
      border: none;
      font: inherit;
    }

    .btn {
      min-height: 48px;
      padding: 0 20px;
      border-radius: 999px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }

    .btn:hover { transform: scale(1.04); }
    .btn:disabled {
      opacity: 0.65;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    .btn-primary {
      background: var(--brand);
      color: #fff;
      box-shadow: 0 16px 32px rgba(13, 111, 212, 0.2);
    }

    .btn-secondary {
      background: #fff;
      color: var(--brand-dark);
      border: 1px solid var(--line);
    }

    .modal-layer {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      background: rgba(16, 35, 69, 0.42);
      backdrop-filter: blur(8px);
      z-index: 999;
    }

    .modal-layer.show { display: flex; }

    .modal {
      width: min(520px, 100%);
      background: #fff;
      border-radius: 24px;
      padding: 26px;
      box-shadow: 0 26px 70px rgba(16, 35, 69, 0.22);
    }

    .modal h3 {
      margin: 0 0 12px;
      font-size: 1.36rem;
    }

    .modal p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }

    .modal-meta {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 16px;
      background: var(--surface-soft);
      color: var(--brand-dark);
      font-weight: 600;
      font-size: 0.92rem;
    }

    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 20px;
    }

    @media (max-width: 760px) {
      .grid, .summary-grid, .progress {
        grid-template-columns: 1fr;
      }

      .topbar, .actions {
        flex-direction: column;
        align-items: stretch;
      }

      .actions-group, .btn {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="survey-shell">
      <div class="survey-header">
        <h1>Course Feedback Survey</h1>
        <p>Please complete the form below and submit your review.</p>
      </div>

      <div class="survey-card">
        <div class="topbar">
          <h2>Survey Form</h2>
          <div class="status-pill" id="statusPill">Step 1 of 3</div>
        </div>

        <div class="progress">
          <div class="progress-step active"><small>Step 1</small><strong>About You</strong></div>
          <div class="progress-step"><small>Step 2</small><strong>Course Review</strong></div>
          <div class="progress-step"><small>Step 3</small><strong>Submit</strong></div>
        </div>

        <div id="banner" class="banner"></div>

        <form id="surveyForm">
          <section class="step-panel active">
            <div class="section">
              <h3>Student Details</h3>
              <div class="grid">
                <div class="field">
                  <label for="nickname">Nickname</label>
                  <input id="nickname" name="nickname" placeholder="Enter a nickname" />
                </div>
                <div class="field">
                  <label for="major">Major</label>
                  <input id="major" name="major" placeholder="Enter your major" />
                </div>
              </div>
            </div>

            <div class="section">
              <h3>Course Information</h3>
              <div class="grid">
                <div class="field">
                  <label for="course_code">Course Code</label>
                  <input id="course_code" name="course_code" placeholder="For example: COMP1511" />
                </div>
                <div class="field">
                  <label for="study_year">Study Year</label>
                  <select id="study_year" name="study_year"></select>
                </div>
                <div class="field">
                  <label for="study_term">Study Term</label>
                  <select id="study_term" name="study_term"></select>
                </div>
                <div class="field">
                  <label for="academic_level">Academic Level</label>
                  <select id="academic_level" name="academic_level"></select>
                </div>
              </div>
            </div>
          </section>

          <section class="step-panel">
            <div class="section">
              <h3>Course Ratings</h3>
              <div class="grid">
                <div class="field">
                  <label>Difficulty</label>
                  <p class="hint">1 = easiest, 5 = hardest</p>
                  <div class="rating-group" data-rating="difficulty_rating"></div>
                </div>
                <div class="field">
                  <label>Workload</label>
                  <p class="hint">1 = lightest, 5 = heaviest</p>
                  <div class="rating-group" data-rating="workload_rating"></div>
                </div>
                <div class="field">
                  <label>Teaching Quality</label>
                  <p class="hint">1 = worst, 5 = best</p>
                  <div class="rating-group" data-rating="teaching_rating"></div>
                </div>
                <div class="field">
                  <label>Recommendation</label>
                  <p class="hint">1 = least likely, 5 = most likely</p>
                  <div class="rating-group" data-rating="recommendation_rating"></div>
                </div>
              </div>
            </div>

            <div class="section">
              <h3>Written Review</h3>
              <div class="field-full">
                <label for="review_text">Course Review</label>
                <textarea id="review_text" name="review_text" placeholder="Enter your review here."></textarea>
              </div>
            </div>
          </section>

          <section class="step-panel">
            <div class="section">
              <h3>Contact</h3>
              <div class="grid">
                <div class="field-full">
                  <label for="contact_email">Email Address</label>
                  <input id="contact_email" name="contact_email" type="email" placeholder="name@example.com" />
                </div>
              </div>
            </div>

            <div class="section">
              <h3>Summary</h3>
              <div class="summary-grid">
                <div class="summary-tile">
                  <strong>Course</strong>
                  <span id="summaryCourse">Not filled in yet</span>
                </div>
                <div class="summary-tile">
                  <strong>Ratings</strong>
                  <span id="summaryRatings">Not filled in yet</span>
                </div>
                <div class="summary-tile">
                  <strong>Review</strong>
                  <span id="summaryReview">Not filled in yet</span>
                </div>
                <div class="summary-tile">
                  <strong>Moderation</strong>
                  <span>Joint text + scores model</span>
                </div>
              </div>
            </div>
          </section>

          <div class="actions">
            <div class="actions-group">
              <button type="button" class="btn btn-secondary" id="prevBtn">Previous</button>
            </div>
            <div class="actions-group">
              <button type="button" class="btn btn-secondary" id="clearBtn">Clear Step</button>
              <button type="button" class="btn btn-primary" id="nextBtn">Next</button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </div>

  <div class="modal-layer" id="modalLayer">
    <div class="modal">
      <h3 id="modalTitle"></h3>
      <p id="modalBody"></p>
      <div class="modal-meta" id="modalMeta"></div>
      <div class="modal-actions" id="modalActions"></div>
    </div>
  </div>

  <script>
    const state = {
      step: 0,
      checking: false,
      submitting: false,
      ratings: {
        difficulty_rating: 0,
        workload_rating: 0,
        teaching_rating: 0,
        recommendation_rating: 0,
      },
    };

    const form = document.getElementById("surveyForm");
    const banner = document.getElementById("banner");
    const statusPill = document.getElementById("statusPill");
    const panels = Array.from(document.querySelectorAll(".step-panel"));
    const progressSteps = Array.from(document.querySelectorAll(".progress-step"));
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const clearBtn = document.getElementById("clearBtn");
    const modalLayer = document.getElementById("modalLayer");
    const modalTitle = document.getElementById("modalTitle");
    const modalBody = document.getElementById("modalBody");
    const modalMeta = document.getElementById("modalMeta");
    const modalActions = document.getElementById("modalActions");

    function setupSelectOptions() {
      const yearSelect = document.getElementById("study_year");
      const termSelect = document.getElementById("study_term");
      const levelSelect = document.getElementById("academic_level");
      const currentYear = new Date().getFullYear();

      yearSelect.innerHTML = '<option value="">Select a year</option>';
      for (let year = currentYear; year >= currentYear - 4; year -= 1) {
        yearSelect.innerHTML += '<option value="' + year + '">' + year + '</option>';
      }

      termSelect.innerHTML = [
        '<option value="">Select a term</option>',
        '<option value="T1">T1</option>',
        '<option value="T2">T2</option>',
        '<option value="T3">T3</option>',
        '<option value="Summer">Summer</option>'
      ].join("");

      levelSelect.innerHTML = [
        '<option value="">Select a level</option>',
        '<option value="undergraduate">Undergraduate</option>',
        '<option value="postgraduate">Postgraduate</option>',
        '<option value="doctorate">Doctorate</option>'
      ].join("");
    }

    function showBanner(message, kind) {
      banner.textContent = message || "";
      banner.className = "banner show " + kind;
    }

    function clearBanner() {
      banner.textContent = "";
      banner.className = "banner";
    }

    function normalizeCourseCode(value) {
      return String(value || "").trim().toUpperCase().replace(/\\s+/g, "");
    }

    function emailValid(value) {
      return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(String(value || "").trim());
    }

    function getPayload() {
      const data = Object.fromEntries(new FormData(form).entries());
      return {
        nickname: String(data.nickname || "").trim(),
        major: String(data.major || "").trim(),
        course_code: normalizeCourseCode(data.course_code),
        study_year: String(data.study_year || "").trim(),
        study_term: String(data.study_term || "").trim(),
        academic_level: String(data.academic_level || "").trim(),
        difficulty_rating: state.ratings.difficulty_rating,
        workload_rating: state.ratings.workload_rating,
        teaching_rating: state.ratings.teaching_rating,
        recommendation_rating: state.ratings.recommendation_rating,
        review_text: String(data.review_text || "").trim(),
        contact_email: String(data.contact_email || "").trim(),
      };
    }

    function setStep(step) {
      state.step = step;
      panels.forEach((panel, index) => {
        panel.classList.toggle("active", index === step);
      });
      progressSteps.forEach((item, index) => {
        item.classList.toggle("active", index <= step);
      });
      statusPill.textContent = "Step " + (step + 1) + " of 3";
      prevBtn.style.visibility = step === 0 ? "hidden" : "visible";
      nextBtn.textContent = step === 2
        ? (state.submitting ? "Submitting..." : state.checking ? "Checking..." : "Submit")
        : "Next";
      updateSummary();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function validateStep(step) {
      const payload = getPayload();
      clearBanner();

      if (step === 0) {
        if (!payload.course_code) {
          showBanner("Please enter the course code.", "error");
          return false;
        }
        if (!payload.study_year || !payload.study_term || !payload.academic_level) {
          showBanner("Please complete the course information.", "error");
          return false;
        }
      }

      if (step === 1) {
        if (Object.values(state.ratings).some((value) => value < 1 || value > 5)) {
          showBanner("Please complete all four rating fields.", "error");
          return false;
        }
        if (!payload.review_text) {
          showBanner("Please enter the written review.", "error");
          return false;
        }
      }

      if (step === 2) {
        if (!payload.contact_email) {
          showBanner("Please enter the email address.", "error");
          return false;
        }
        if (!emailValid(payload.contact_email)) {
          showBanner("Please enter a valid email address.", "error");
          return false;
        }
      }

      return true;
    }

    function updateSummary() {
      const payload = getPayload();
      document.getElementById("summaryCourse").textContent = payload.course_code
        ? [payload.course_code, payload.study_year, payload.study_term].filter(Boolean).join(" | ")
        : "Not filled in yet";

      const ratings = [
        ["Difficulty", state.ratings.difficulty_rating],
        ["Workload", state.ratings.workload_rating],
        ["Teaching", state.ratings.teaching_rating],
        ["Recommendation", state.ratings.recommendation_rating],
      ].filter((item) => item[1] > 0);

      document.getElementById("summaryRatings").textContent = ratings.length
        ? ratings.map((item) => item[0] + " " + item[1] + "/5").join(" | ")
        : "Not filled in yet";

      document.getElementById("summaryReview").textContent = payload.review_text
        ? payload.review_text.slice(0, 140) + (payload.review_text.length > 140 ? "..." : "")
        : "Not filled in yet";
    }

    function makeButton(label, kind, onClick) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn " + kind;
      button.textContent = label;
      button.addEventListener("click", onClick);
      return button;
    }

    function openModal(config) {
      modalTitle.textContent = config.title;
      modalBody.textContent = config.body;
      modalMeta.textContent = config.meta || "";
      modalMeta.style.display = config.meta ? "block" : "none";
      modalActions.innerHTML = "";
      (config.actions || []).forEach((action) => {
        modalActions.appendChild(makeButton(action.label, action.kind, action.onClick));
      });
      modalLayer.classList.add("show");
    }

    function closeModal() {
      modalLayer.classList.remove("show");
    }

    function clearCurrentStep() {
      clearBanner();
      if (state.step === 0) {
        ["nickname", "major", "course_code", "study_year", "study_term", "academic_level"].forEach((name) => {
          const field = form.elements.namedItem(name);
          if (field) field.value = "";
        });
      } else if (state.step === 1) {
        Object.keys(state.ratings).forEach((key) => {
          state.ratings[key] = 0;
        });
        document.querySelectorAll(".rating-chip").forEach((chip) => chip.classList.remove("active"));
        const review = form.elements.namedItem("review_text");
        if (review) review.value = "";
      } else {
        const email = form.elements.namedItem("contact_email");
        if (email) email.value = "";
      }
      updateSummary();
    }

    function renderRatingGroups() {
      document.querySelectorAll(".rating-group").forEach((group) => {
        const key = group.getAttribute("data-rating");
        const row = document.createElement("div");
        row.className = "rating-row";

        for (let value = 1; value <= 5; value += 1) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "rating-chip";
          button.textContent = String(value);
          button.addEventListener("click", () => {
            state.ratings[key] = value;
            row.querySelectorAll(".rating-chip").forEach((chip, index) => {
              chip.classList.toggle("active", index + 1 === value);
            });
            updateSummary();
          });
          row.appendChild(button);
        }

        group.appendChild(row);
      });
    }

    async function submitSurvey() {
      const payload = getPayload();
      const response = await fetch("/api/course-surveys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Unable to submit the survey.");
      }
      form.reset();
      Object.keys(state.ratings).forEach((key) => {
        state.ratings[key] = 0;
      });
      document.querySelectorAll(".rating-chip").forEach((chip) => chip.classList.remove("active"));
      showBanner("Survey submitted successfully.", "success");
      setStep(0);
      updateSummary();
    }

    async function runModeration() {
      if (!validateStep(2)) {
        return;
      }

      const payload = getPayload();
      state.checking = true;
      nextBtn.disabled = true;
      setStep(2);

      try {
        const response = await fetch("/api/course-surveys/moderation-check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || "Unable to run moderation.");
        }

        const meta = "Risk score: " + Math.round(Number(result.probability_hate || 0) * 100) + "%";

        if (result.moderation_level === "blocked") {
          openModal({
            title: "Please revise your review",
            body: "This review has been blocked by the moderation model. Please revise the text before submitting.",
            meta,
            actions: [
              {
                label: "Edit Review",
                kind: "btn-primary",
                onClick: () => {
                  closeModal();
                  setStep(1);
                },
              },
            ],
          });
          showBanner("The review has been blocked and needs revision.", "error");
          return;
        }

        if (result.moderation_level === "warning") {
          openModal({
            title: "Please review your text",
            body: "This review falls into the warning range. You may revise the text or continue with the submission.",
            meta,
            actions: [
              {
                label: "Edit Review",
                kind: "btn-secondary",
                onClick: () => {
                  closeModal();
                  setStep(1);
                },
              },
              {
                label: "Continue",
                kind: "btn-primary",
                onClick: async () => {
                  closeModal();
                  state.submitting = true;
                  nextBtn.disabled = true;
                  setStep(2);
                  try {
                    await submitSurvey();
                  } catch (error) {
                    showBanner(error.message || "Unable to submit the survey.", "error");
                  } finally {
                    state.submitting = false;
                    state.checking = false;
                    nextBtn.disabled = false;
                    setStep(state.step);
                  }
                },
              },
            ],
          });
          showBanner("The review is in the warning range.", "warning");
          return;
        }

        openModal({
          title: "Review accepted",
          body: "This review has passed the moderation check. Continue to submit the survey?",
          meta,
          actions: [
            {
              label: "Edit Review",
              kind: "btn-secondary",
              onClick: () => {
                closeModal();
              },
            },
            {
              label: "Submit",
              kind: "btn-primary",
              onClick: async () => {
                closeModal();
                state.submitting = true;
                nextBtn.disabled = true;
                setStep(2);
                try {
                  await submitSurvey();
                } catch (error) {
                  showBanner(error.message || "Unable to submit the survey.", "error");
                } finally {
                  state.submitting = false;
                  state.checking = false;
                  nextBtn.disabled = false;
                  setStep(state.step);
                }
              },
            },
          ],
        });
      } catch (error) {
        showBanner(error.message || "Unable to run moderation.", "error");
      } finally {
        state.checking = false;
        nextBtn.disabled = false;
        setStep(state.step);
      }
    }

    prevBtn.addEventListener("click", () => {
      clearBanner();
      if (state.step > 0) setStep(state.step - 1);
    });

    nextBtn.addEventListener("click", async () => {
      clearBanner();
      if (state.step < 2) {
        if (validateStep(state.step)) setStep(state.step + 1);
        return;
      }
      await runModeration();
    });

    clearBtn.addEventListener("click", clearCurrentStep);

    form.addEventListener("input", updateSummary);

    modalLayer.addEventListener("click", (event) => {
      if (event.target === modalLayer) closeModal();
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal();
    });

    renderRatingGroups();
    setupSelectOptions();
    updateSummary();
    setStep(0);
  </script>
</body>
</html>
"""


class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def normalize_score(value: float, mean: float, std: float) -> float:
    safe_std = std if std else 1.0
    return (float(value) - float(mean)) / safe_std


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class JointModerationModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.encoder = AutoModel.from_pretrained(MODEL_DIR).to(self.device)
        self.encoder.eval()

        state_dict = torch.load(CLASSIFIER_HEAD_PATH, map_location=self.device)
        hidden_dim = int(state_dict["net.0.weight"].shape[0])
        input_dim = int(state_dict["net.0.weight"].shape[1])
        self.classifier = FusionClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0).to(self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

    def predict(self, payload: Dict[str, object]) -> Dict[str, object]:
        review_text = str(payload.get("review_text") or "").strip()
        if not review_text:
            raise ValueError("review_text is required")

        raw_scores = {
            "difficulty": float(payload.get("difficulty_rating") or 0.0),
            "workload": float(payload.get("workload_rating") or 0.0),
            "teaching_quality": float(payload.get("teaching_rating") or 0.0),
            "recommendation": float(payload.get("recommendation_rating") or 0.0),
        }

        for name, value in raw_scores.items():
            if value < 1 or value > 5:
                raise ValueError(f"{name} must be between 1 and 5")

        with torch.no_grad():
            encodings = self.tokenizer(
                [review_text],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = self.encoder(
                input_ids=encodings["input_ids"].to(self.device),
                attention_mask=encodings["attention_mask"].to(self.device),
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu()

            normalized_scores = [
                normalize_score(raw_scores[name], SCORE_MEANS[name], SCORE_STDS[name])
                for name in SCORE_COLUMNS
            ]
            score_tensor = torch.tensor([normalized_scores], dtype=torch.float32)
            features = torch.cat([cls_embedding, score_tensor], dim=1).to(self.device)

            logit = self.classifier(features).squeeze().item()
            probability_hate = float(torch.sigmoid(torch.tensor(logit)).item())

        if probability_hate >= BLOCK_THRESHOLD:
            moderation_level = "blocked"
            label = "hate"
            is_hate_speech = True
            is_warning = False
            is_safe = False
            message = "The review is predicted as hate speech. Please revise it before submitting."
        elif probability_hate >= WARNING_THRESHOLD:
            moderation_level = "warning"
            label = "warning"
            is_hate_speech = False
            is_warning = True
            is_safe = False
            message = "The review is in the warning range. Please review it before submitting."
        else:
            moderation_level = "normal"
            label = "normal"
            is_hate_speech = False
            is_warning = False
            is_safe = True
            message = "The review is predicted as normal, non-hate speech."

        return {
            "moderation_level": moderation_level,
            "warning_threshold": WARNING_THRESHOLD,
            "block_threshold": BLOCK_THRESHOLD,
            "is_hate_speech": is_hate_speech,
            "is_warning": is_warning,
            "is_safe": is_safe,
            "label": label,
            "confidence": round(probability_hate if moderation_level != "normal" else 1.0 - probability_hate, 4),
            "probability_hate": round(probability_hate, 4),
            "probability_normal": round(1.0 - probability_hate, 4),
            "source": "joint_model_backend",
            "message": message,
            "model_name": MODEL_NAME,
        }


def append_submission(payload: Dict[str, object]) -> Dict[str, object]:
    ensure_dir(SUBMISSIONS_PATH.parent)
    record = {
        "submission_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with SUBMISSIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


MODEL = JointModerationModel()


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "FinalCourseSurveyDemo/1.0"

    def _send_html(self, status: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(200, render_page())
            return
        self._send_json(404, {"detail": f"Route not found: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"detail": str(exc)})
            return

        if parsed.path == "/api/course-surveys/moderation-check":
            try:
                result = MODEL.predict(payload)
            except ValueError as exc:
                self._send_json(400, {"detail": str(exc)})
                return
            self._send_json(200, result)
            return

        if parsed.path == "/api/course-surveys":
            course_code = str(payload.get("course_code") or "").strip().upper().replace(" ", "")
            if not course_code:
                self._send_json(400, {"detail": "course_code is required"})
                return

            stored = append_submission({
                **payload,
                "course_code": course_code,
                "status": "submitted",
            })
            self._send_json(201, {
                "survey_id": stored["submission_id"],
                "status": stored["status"],
                "created_at": stored["created_at"],
            })
            return

        self._send_json(404, {"detail": f"Route not found: {parsed.path}"})


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    print(f"Course survey demo listening on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final course survey demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--predict-review", default="")
    parser.add_argument("--difficulty", type=float, default=3)
    parser.add_argument("--workload", type=float, default=3)
    parser.add_argument("--teaching", type=float, default=3)
    parser.add_argument("--recommendation", type=float, default=3)
    args = parser.parse_args()

    if args.predict_review:
        result = MODEL.predict({
            "review_text": args.predict_review,
            "difficulty_rating": args.difficulty,
            "workload_rating": args.workload,
            "teaching_rating": args.teaching,
            "recommendation_rating": args.recommendation,
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
