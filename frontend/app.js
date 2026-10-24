// CollegeMate AI frontend: plain JavaScript, no build step.
//
// Security note: everything coming from the server (answers, file names, snippets) is
// inserted with `textContent`, never `innerHTML`, so model output or PDF text can
// never be interpreted as HTML.

"use strict";

// ------------------------------------------------------------------ client identity
// There is no login in this version. The browser creates a random id and sends it with
// every request; the backend scopes documents and chat history to it. This is NOT real
// security (see README), it only keeps different browsers' data apart.
function getClientId() {
  let id = null;
  try { id = localStorage.getItem("collegemate_client_id"); } catch (e) { /* storage blocked */ }
  if (!id) {
    id = (crypto.randomUUID && crypto.randomUUID()) ||
         Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, "0")).join("");
    try { localStorage.setItem("collegemate_client_id", id); } catch (e) { /* ignore */ }
  }
  return id;
}
const CLIENT_ID = getClientId();

// ------------------------------------------------------------------ DOM references
const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const formEl = $("chat-form");
const inputEl = $("message-input");
const sendBtn = $("send-btn");
const errorBanner = $("error-banner");
const fileInput = $("file-input");
const uploadLabel = document.querySelector(".upload-btn");
const uploadStatus = $("upload-status");
const docList = $("doc-list");
const docEmpty = $("doc-empty");

let chatBusy = false;

// ------------------------------------------------------------------ API helper
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "X-Client-ID": CLIENT_ID, ...(options.headers || {}) },
  });
  let data = null;
  try { data = await response.json(); } catch (e) { /* non-JSON body */ }
  if (!response.ok) {
    const message = data && data.error && data.error.message
      ? data.error.message
      : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

// ------------------------------------------------------------------ small DOM helpers
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}
function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ------------------------------------------------------------------ chat rendering
const TYPE_LABELS = {
  document_grounded: "From your documents",
  general: "General answer (not from documents)",
  insufficient_evidence: "Not enough information in documents",
  conflicting_information: "Documents disagree",
};

function addWelcome() {
  const box = el("div", "message welcome");
  box.appendChild(el("div", "text",
    "Hi! Upload your college PDFs, then ask things like “What is the minimum attendance?” " +
    "You can also just chat."));
  messagesEl.appendChild(box);
}

function addMessage(role, text, extras = {}) {
  const box = el("div", `message ${role}`);

  if (role === "assistant" && extras.answerType && TYPE_LABELS[extras.answerType]) {
    box.appendChild(el("span", `tag ${extras.answerType}`, TYPE_LABELS[extras.answerType]));
    box.appendChild(document.createElement("br"));
  }
  box.appendChild(el("div", "text", text));

  (extras.warnings || []).forEach((warning) => box.appendChild(el("div", "warning", "⚠ " + warning)));

  if (extras.sources && extras.sources.length) {
    const wrap = el("div", "sources");
    wrap.appendChild(el("div", "sources-title", "Sources"));
    extras.sources.forEach((source) => {
      const item = el("div", "source");
      const page = source.page != null ? ` — page ${source.page}` : "";
      item.appendChild(el("div", "source-name", `📄 ${source.document_name}${page}`));
      item.appendChild(el("div", "source-snippet", `“${source.snippet}…”`));
      wrap.appendChild(item);
    });
    box.appendChild(wrap);
  }

  messagesEl.appendChild(box);
  scrollToBottom();
  return box;
}

function addTypingIndicator() {
  const box = el("div", "message assistant");
  const dots = el("div", "typing");
  dots.append(el("span"), el("span"), el("span"));
  box.appendChild(dots);
  messagesEl.appendChild(box);
  scrollToBottom();
  return box;
}

function setChatBusy(busy) {
  chatBusy = busy;
  sendBtn.disabled = busy;
  inputEl.disabled = busy;
  if (!busy) inputEl.focus();
}

async function sendMessage(text) {
  clearError();
  addMessage("user", text);
  const typing = addTypingIndicator();
  setChatBusy(true);
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    typing.remove();
    addMessage("assistant", data.answer, {
      answerType: data.answer_type,
      sources: data.sources,
      warnings: data.warnings,
    });
  } catch (error) {
    typing.remove();
    showError(error.message);
  } finally {
    setChatBusy(false);
  }
}

async function loadHistory() {
  messagesEl.replaceChildren();
  try {
    const data = await api("/api/chat/history");
    if (!data.messages.length) { addWelcome(); return; }
    data.messages.forEach((m) => addMessage(m.role, m.content, { answerType: m.answer_type }));
  } catch (error) {
    addWelcome();
    showError(error.message);
  }
}

// ------------------------------------------------------------------ documents
function renderDocuments(documents) {
  docList.replaceChildren();
  docEmpty.hidden = documents.length > 0;

  documents.forEach((doc) => {
    const item = el("li", "doc-item");
    item.appendChild(el("div", "doc-name", doc.filename));

    const meta = el("div", "doc-meta");
    meta.appendChild(el("span", `badge ${doc.status}`, doc.status));
    if (doc.status === "ready") {
      meta.appendChild(document.createTextNode(`${doc.page_count} pages · ${doc.chunk_count} chunks`));
    }
    item.appendChild(meta);

    if (doc.error) item.appendChild(el("div", "doc-error", doc.error));

    const actions = el("div", "doc-actions");
    const reprocess = el("button", "link-btn", "Reprocess");
    reprocess.type = "button";
    reprocess.addEventListener("click", () => reprocessDocument(doc.id));
    const remove = el("button", "link-btn danger", "Delete");
    remove.type = "button";
    remove.addEventListener("click", () => deleteDocument(doc.id, doc.filename));
    actions.append(reprocess, remove);
    item.appendChild(actions);

    docList.appendChild(item);
  });
}

async function loadDocuments() {
  try {
    const data = await api("/api/documents");
    renderDocuments(data.documents);
  } catch (error) {
    setUploadStatus(error.message, "error");
  }
}

function setUploadStatus(text, mode) {
  uploadStatus.textContent = text;
  uploadStatus.className = "status" + (mode ? " " + mode : "");
}

function setUploadBusy(busy) {
  uploadLabel.classList.toggle("disabled", busy);
  fileInput.disabled = busy;
}

async function uploadFiles(files) {
  setUploadBusy(true);
  const problems = [];
  for (const [index, file] of files.entries()) {
    setUploadStatus(`Processing ${file.name} (${index + 1}/${files.length})…`, "busy");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const doc = await api("/api/documents/upload", { method: "POST", body: formData });
      if (doc.status === "failed") problems.push(`${doc.filename}: ${doc.error}`);
    } catch (error) {
      problems.push(`${file.name}: ${error.message}`);
    }
    await loadDocuments();
  }
  setUploadBusy(false);
  if (problems.length) setUploadStatus(problems.join("\n"), "error");
  else setUploadStatus(`Uploaded ${files.length} file(s) successfully.`);
}

async function reprocessDocument(id) {
  setUploadBusy(true);
  setUploadStatus("Reprocessing…", "busy");
  try {
    const doc = await api(`/api/documents/${id}/reprocess`, { method: "POST" });
    if (doc.status === "failed") setUploadStatus(`${doc.filename}: ${doc.error}`, "error");
    else setUploadStatus("Document reprocessed.");
  } catch (error) {
    setUploadStatus(error.message, "error");
  }
  setUploadBusy(false);
  await loadDocuments();
}

async function deleteDocument(id, name) {
  if (!confirm(`Delete “${name}”? It will no longer be used to answer questions.`)) return;
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    setUploadStatus("Document deleted.");
  } catch (error) {
    setUploadStatus(error.message, "error");
  }
  await loadDocuments();
}

// ------------------------------------------------------------------ events
formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text || chatBusy) return;
  inputEl.value = "";
  autoGrow();
  sendMessage(text);
});

// Enter sends, Shift+Enter inserts a new line.
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}
inputEl.addEventListener("input", autoGrow);

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files);
  fileInput.value = ""; // allow selecting the same file again later
  if (files.length) uploadFiles(files);
});

$("clear-btn").addEventListener("click", async () => {
  if (!confirm("Clear the whole conversation?")) return;
  clearError();
  try {
    await api("/api/chat/history", { method: "DELETE" });
    messagesEl.replaceChildren();
    addWelcome();
  } catch (error) {
    showError(error.message);
  }
});

// ------------------------------------------------------------------ start
loadDocuments();
loadHistory();
