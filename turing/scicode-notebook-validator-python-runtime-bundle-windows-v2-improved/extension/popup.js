const HOST_NAME = "com.scicode.validator.native";

let latestReportText = "";
let latestBaseName = "scicode_native_validation";

function setStatus(message, isError = false) {
  const node = document.getElementById("status");
  node.textContent = message;
  node.style.color = isError ? "#b91c1c" : "#0f172a";
}

function setOutput(text) {
  document.getElementById("output").textContent = text || "";
}

function appendOutput(text) {
  const node = document.getElementById("output");
  node.textContent += `${text}\n`;
}

function setButtonsDisabled(disabled) {
  document.getElementById("validateBtn").disabled = disabled;
  document.getElementById("validateCurrentBtn").disabled = disabled;
  document.getElementById("downloadReportBtn").disabled = disabled || !latestReportText;
}

function clearRenderedPanels() {
  document.getElementById("kpiGrid").classList.add("hidden");
  document.getElementById("kpiOverall").textContent = "-";
  document.getElementById("kpiIssues").textContent = "0";
  document.getElementById("kpiFails").textContent = "0";
  document.getElementById("kpiPasses").textContent = "0";
  document.getElementById("summaryCard").className = "summary hidden";
  document.getElementById("summaryCard").textContent = "";
  document.getElementById("issuesCard").classList.add("hidden");
  document.getElementById("resultsCard").classList.add("hidden");
  document.getElementById("issuesList").innerHTML = "";
  document.getElementById("resultsList").innerHTML = "";
}

function appendListItem(listNode, text, className) {
  const li = document.createElement("li");
  li.className = className;
  li.textContent = text;
  listNode.appendChild(li);
}

function getIssueLines(response) {
  if (Array.isArray(response.issues_preview)) {
    return response.issues_preview;
  }
  if (Array.isArray(response.issues)) {
    return response.issues;
  }
  return [];
}

function getResultLines(response) {
  if (Array.isArray(response.results_preview)) {
    return response.results_preview;
  }
  if (Array.isArray(response.results)) {
    return response.results;
  }
  return [];
}

function setKpis(overall, issueCount, failLines, passLines) {
  document.getElementById("kpiGrid").classList.remove("hidden");
  document.getElementById("kpiOverall").textContent = overall;
  document.getElementById("kpiIssues").textContent = String(issueCount);
  document.getElementById("kpiFails").textContent = String(failLines);
  document.getElementById("kpiPasses").textContent = String(passLines);
}

function renderResponsePanels(response) {
  clearRenderedPanels();
  const summary = document.getElementById("summaryCard");
  const issuesCard = document.getElementById("issuesCard");
  const issuesList = document.getElementById("issuesList");
  const resultsCard = document.getElementById("resultsCard");
  const resultsList = document.getElementById("resultsList");

  const issueLines = getIssueLines(response);
  const resultLines = getResultLines(response);
  const issueCount = Number(response.issue_count ?? issueLines.length);
  const failLines = Number(
    response.fail_count ?? resultLines.filter((x) => String(x).trim().startsWith("FAIL ")).length
  );
  const passLines = Number(
    response.pass_count ?? resultLines.filter((x) => String(x).trim().startsWith("PASS ")).length
  );
  const overall = String(response.overall_status || "UNKNOWN");
  const previewTruncated = Boolean(response.preview_truncated);

  setKpis(overall, issueCount, failLines, passLines);

  summary.className = `summary ${overall === "PASS" ? "pass" : "fail"}`;
  summary.textContent = previewTruncated
    ? "Preview is truncated for reliable native messaging. Full report is loaded from backend logs below."
    : "Validation preview loaded successfully.";

  if (issueLines.length > 0) {
    issuesCard.classList.remove("hidden");
    for (const issue of issueLines) {
      appendListItem(issuesList, issue, "fail");
    }
    if (issueLines.length < issueCount) {
      appendListItem(issuesList, `Showing ${issueLines.length} of ${issueCount} issue lines.`, "neutral");
    }
  }

  if (resultLines.length > 0) {
    resultsCard.classList.remove("hidden");
    for (const line of resultLines) {
      const text = String(line);
      let cls = "neutral";
      if (text.trim().startsWith("FAIL ")) {
        cls = "fail";
      } else if (text.trim().startsWith("PASS ")) {
        cls = "pass";
      }
      appendListItem(resultsList, text, cls);
    }
    const resultCount = Number(response.result_count ?? resultLines.length);
    if (resultLines.length < resultCount) {
      appendListItem(resultsList, `Showing ${resultLines.length} of ${resultCount} result lines.`, "neutral");
    }
  }
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractFileId(url) {
  const patterns = [
    /colab\.research\.google\.com\/drive\/([a-zA-Z0-9_-]+)/,
    /drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)/,
    /drive\.google\.com\/open\?id=([a-zA-Z0-9_-]+)/,
    /drive\.google\.com\/uc\?id=([a-zA-Z0-9_-]+)/,
    /docs\.google\.com\/uc\?id=([a-zA-Z0-9_-]+)/
  ];
  for (const pattern of patterns) {
    const match = String(url || "").match(pattern);
    if (match) {
      return match[1];
    }
  }
  return null;
}

function parseNotebookJson(raw) {
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || !("nbformat" in parsed) || !Array.isArray(parsed.cells)) {
    throw new Error("Fetched content is not a valid .ipynb notebook.");
  }
  return parsed;
}

function buildCompactNotebookPayload(notebook) {
  const compact = {
    nbformat: notebook?.nbformat ?? 4,
    nbformat_minor: notebook?.nbformat_minor ?? 0,
    metadata: notebook?.metadata ?? {},
    cells: []
  };

  const cells = Array.isArray(notebook?.cells) ? notebook.cells : [];
  compact.cells = cells.map((cell) => {
    const base = {
      cell_type: cell?.cell_type || "markdown",
      metadata: cell?.metadata ?? {},
      source: Array.isArray(cell?.source) ? cell.source : [String(cell?.source ?? "")]
    };
    // Drop heavy output payloads; validator checks source structure/content.
    if (base.cell_type === "code") {
      base.execution_count = null;
      base.outputs = [];
    }
    return base;
  });
  return compact;
}

function withTimeout(promise, ms, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

async function fetchText(url) {
  const response = await fetch(url, {
    method: "GET",
    redirect: "follow",
    credentials: "include"
  });
  const text = await response.text();
  return { ok: response.ok, status: response.status, text };
}

async function fetchBackendLogText(logPath) {
  if (!logPath) {
    return "";
  }
  const chunks = [];
  let offset = 0;
  const chunkSize = 180000;
  const maxBytes = 3_000_000;

  while (offset < maxBytes) {
    const chunkResponse = await withTimeout(
      connectNativeAndSend({
        action: "read_log_chunk",
        log_path: logPath,
        offset,
        chunk_size: chunkSize
      }),
      30000,
      "Timed out while streaming backend report log."
    );

    if (!chunkResponse || chunkResponse.ok !== true) {
      throw new Error(chunkResponse?.error || "Failed to stream backend report log.");
    }

    const content = String(chunkResponse.content || "");
    chunks.push(content);
    offset = Number(chunkResponse.next_offset ?? offset + content.length);
    if (chunkResponse.eof) {
      break;
    }
  }

  if (offset >= maxBytes) {
    chunks.push("\n\n[Report truncated in popup at 3MB. Download report to view full content.]");
  }
  return chunks.join("");
}

async function fetchNotebookFromUrl(url) {
  const trimmed = url.trim();
  const errors = [];

  if (trimmed.toLowerCase().endsWith(".ipynb")) {
    const direct = await fetchText(trimmed);
    if (!direct.ok) {
      throw new Error(`Direct fetch failed (${direct.status}).`);
    }
    return parseNotebookJson(direct.text);
  }

  const fileId = extractFileId(trimmed);
  if (!fileId) {
    throw new Error("Could not extract Google Drive file id from URL.");
  }

  const attempts = [
    `https://drive.google.com/uc?export=download&id=${fileId}`,
    `https://drive.google.com/uc?id=${fileId}&export=download`,
    `https://docs.google.com/uc?export=download&id=${fileId}`,
    `https://drive.usercontent.google.com/download?id=${fileId}&export=download`,
    `https://colab.research.google.com/drive/${fileId}?format=raw`
  ];

  for (const attemptUrl of attempts) {
    try {
      const res = await fetchText(attemptUrl);
      if (!res.ok) {
        errors.push(`${attemptUrl} -> HTTP ${res.status}`);
        continue;
      }
      try {
        return parseNotebookJson(res.text);
      } catch (_err) {
        const tokenMatch =
          res.text.match(/[?&]confirm=([0-9A-Za-z_]+)/) ||
          res.text.match(/name="confirm"\s+value="([0-9A-Za-z_]+)"/);
        if (tokenMatch) {
          const confirmUrl = `https://drive.google.com/uc?export=download&id=${fileId}&confirm=${tokenMatch[1]}`;
          const confirmRes = await fetchText(confirmUrl);
          if (confirmRes.ok) {
            try {
              return parseNotebookJson(confirmRes.text);
            } catch (confirmErr) {
              errors.push(`${confirmUrl} -> invalid notebook (${confirmErr.message})`);
            }
          } else {
            errors.push(`${confirmUrl} -> HTTP ${confirmRes.status}`);
          }
        } else {
          errors.push(`${attemptUrl} -> returned HTML/non-notebook content`);
        }
      }
    } catch (err) {
      errors.push(`${attemptUrl} -> ${err.message}`);
    }
  }

  throw new Error(
    "Could not download notebook from URL.\n" +
      "Ensure you are signed in on this Chrome profile and have notebook access.\n" +
      errors.slice(0, 4).join("\n")
  );
}

function connectNativeAndSend(payload) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let port;
    try {
      port = chrome.runtime.connectNative(HOST_NAME);
    } catch (err) {
      reject(new Error(`Notebook backend host not available: ${err.message}`));
      return;
    }

    const done = (fn, value) => {
      if (settled) {
        return;
      }
      settled = true;
      try {
        port.disconnect();
      } catch (_e) {
        // ignore
      }
      fn(value);
    };

    port.onMessage.addListener((message) => {
      done(resolve, message);
    });

    port.onDisconnect.addListener(() => {
      if (settled) {
        return;
      }
      const errMsg = chrome.runtime.lastError?.message || "Notebook backend host disconnected.";
      done(reject, new Error(errMsg));
    });

    port.postMessage(payload);
  });
}

function formatBackendReport(result) {
  const lines = [];
  lines.push("=".repeat(72));
  lines.push("SCICODE PYTHON VALIDATOR REPORT (NOTEBOOK BACKEND)");
  lines.push("=".repeat(72));
  lines.push(`Overall result: ${result.overall_status || "UNKNOWN"}`);
  lines.push(`Total issues: ${Number(result.issue_count ?? 0)}`);
  lines.push(`Runtime result lines: ${Number(result.result_count ?? 0)}`);
  lines.push(`PASS lines: ${Number(result.pass_count ?? 0)} | FAIL lines: ${Number(result.fail_count ?? 0)}`);
  lines.push("");

  const issueLines = getIssueLines(result);
  if (issueLines.length > 0) {
    lines.push("Issues:");
    for (const issue of issueLines) {
      lines.push(`- ${issue}`);
    }
    if (issueLines.length < Number(result.issue_count ?? issueLines.length)) {
      lines.push(`- ... preview truncated (${issueLines.length}/${result.issue_count})`);
    }
    lines.push("");
  }

  const resultLines = getResultLines(result);
  if (resultLines.length > 0) {
    lines.push("Execution results (validator):");
    for (const line of resultLines) {
      lines.push(`- ${line}`);
    }
    if (resultLines.length < Number(result.result_count ?? resultLines.length)) {
      lines.push(`- ... preview truncated (${resultLines.length}/${result.result_count})`);
    }
    lines.push("");
  }

  if (result.log_path) {
    lines.push(`Log path: ${result.log_path}`);
  }
  return lines.join("\n");
}

function downloadText(filename, text, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(href);
}

async function getActiveTabUrl() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs || tabs.length === 0 || !tabs[0].url) {
    return "";
  }
  return tabs[0].url;
}

function isNotebookPageUrl(url) {
  return /https:\/\/(colab\.research\.google\.com|drive\.google\.com|docs\.google\.com)\//i.test(url || "");
}

async function prefillFromCurrentTab() {
  try {
    const currentValue = document.getElementById("notebookUrl").value.trim();
    if (currentValue) {
      return;
    }
    const tabUrl = await getActiveTabUrl();
    if (isNotebookPageUrl(tabUrl)) {
      document.getElementById("notebookUrl").value = tabUrl;
    }
  } catch (_err) {
    // non-fatal
  }
}

async function validateNotebookByUrl(url) {
  latestReportText = "";
  setButtonsDisabled(true);
  setOutput("");
  clearRenderedPanels();
  setStatus("Fetching notebook...");

  try {
    appendOutput("Step 1/3: Fetch notebook from URL");
    const notebook = await fetchNotebookFromUrl(url);
    const compactNotebook = buildCompactNotebookPayload(notebook);

    appendOutput("Step 2/3: Send notebook to notebook Python backend");
    const runTests = document.getElementById("runTests").checked;
    const response = await withTimeout(
      connectNativeAndSend({
        action: "validate_notebook",
        source_url: url,
        notebook_json: compactNotebook,
        run_tests: Boolean(runTests),
        assert_checks: true
      }),
      1200000,
      "Notebook backend timed out after 20 minutes. Try again after clearing notebook outputs or reducing notebook size."
    );

    if (!response || response.ok !== true) {
      throw new Error(response?.error || "Notebook backend failed.");
    }

    appendOutput("Step 3/3: Load report from backend log");
    const fileId = extractFileId(url) || "colab_notebook";
    latestBaseName = `${fileId}_native_scicode`;
    try {
      latestReportText = await fetchBackendLogText(response.log_path);
    } catch (logErr) {
      appendOutput(`Warning: ${logErr.message}`);
      latestReportText = formatBackendReport(response);
    }
    setOutput(latestReportText);
    renderResponsePanels(response);

    const issueCount = Number(response.issue_count ?? getIssueLines(response).length);
    setStatus(
      issueCount === 0 ? "Validation completed: PASS." : `Validation completed: ${issueCount} issue(s) found.`,
      issueCount > 0
    );
  } catch (err) {
    setStatus(`Validation failed: ${err.message}`, true);
    appendOutput(`Error: ${err.message}`);
  } finally {
    setButtonsDisabled(false);
  }
}

async function onValidateClick() {
  const url = document.getElementById("notebookUrl").value.trim();
  if (!url) {
    setStatus("Paste a Colab/Drive notebook URL or use 'Validate Current Tab'.", true);
    return;
  }
  await validateNotebookByUrl(url);
}

async function onValidateCurrentTabClick() {
  try {
    const tabUrl = await getActiveTabUrl();
    if (!tabUrl) {
      setStatus("Could not read current tab URL.", true);
      return;
    }
    if (!isNotebookPageUrl(tabUrl)) {
      setStatus("Current tab is not a Colab/Drive notebook page.", true);
      return;
    }
    document.getElementById("notebookUrl").value = tabUrl;
    await validateNotebookByUrl(tabUrl);
  } catch (err) {
    setStatus(`Failed to read current tab: ${err.message}`, true);
  }
}

document.getElementById("validateBtn").addEventListener("click", onValidateClick);
document.getElementById("validateCurrentBtn").addEventListener("click", onValidateCurrentTabClick);
document.getElementById("downloadReportBtn").addEventListener("click", () => {
  if (!latestReportText) {
    return;
  }
  downloadText(`${latestBaseName}_report.log`, latestReportText);
});

prefillFromCurrentTab();
