const scenarios = {
  "variant-a": {
    url: "variant-a.html",
    address: "127.0.0.1:8765/variant-a.html",
    viewport: "1280 × 800",
    stage: "wide",
  },
  "variant-b": {
    url: "variant-b.html?notice=0&delay=650",
    address: "127.0.0.1:8765/variant-b.html?notice=0&delay=650",
    viewport: "500 × 800",
    stage: "narrow",
  },
  ambiguous: {
    url: "variant-b.html?notice=0&duplicate_submit=1&delay=0",
    address: "127.0.0.1:8765/variant-b.html?duplicate_submit=1",
    viewport: "500 × 800",
    stage: "narrow",
  },
};

const ui = {
  form: document.querySelector("#parameter-form"),
  run: document.querySelector("#run-button"),
  reset: document.querySelector("#reset-button"),
  frame: document.querySelector("#target-frame"),
  stage: document.querySelector("#browser-stage"),
  status: document.querySelector("#run-status"),
  trace: document.querySelector("#trace-list"),
  state: document.querySelector("#current-state"),
  selector: document.querySelector("#current-selector"),
  resolverTitle: document.querySelector("#resolver-title"),
  matchCount: document.querySelector("#match-count"),
  chips: document.querySelector("#locator-chips"),
  geometry: document.querySelector("#geometry-value"),
  address: document.querySelector("#address-text"),
  viewport: document.querySelector("#viewport-badge"),
  callout: document.querySelector("#target-callout"),
  calloutText: document.querySelector("#target-callout-text"),
};

let selectedScenario = "variant-a";
let procedure;
let running = false;

const normalize = (value) => String(value ?? "").replace(/\s+/g, " ").trim().toLowerCase();
const wait = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));
const animationDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 40 : 380;

function setStatus(value, cssClass = value) {
  ui.status.textContent = value.replaceAll("_", " ");
  ui.status.className = `status-badge ${cssClass}`;
}

function setBusy(value) {
  running = value;
  ui.run.disabled = value;
  ui.reset.disabled = value;
  document.querySelectorAll(".scenario").forEach((button) => { button.disabled = value; });
}

function resetAudit() {
  setStatus("idle");
  ui.state.textContent = "Ready to replay";
  ui.selector.textContent = "fresh DOM only";
  ui.resolverTitle.textContent = "Waiting for a step";
  ui.matchCount.textContent = "— matches";
  ui.matchCount.className = "match-count";
  ui.geometry.textContent = "queried after binding";
  ui.chips.replaceChildren(...["role", "name", "attributes"].map(makeChip));
  ui.callout.hidden = true;
  ui.trace.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty-trace";
  const orbit = document.createElement("span");
  orbit.className = "empty-orbit";
  const copy = document.createElement("p");
  copy.textContent = "Choose a scenario and run the procedure to inspect every decision.";
  empty.append(orbit, copy);
  ui.trace.append(empty);
}

function makeChip(value) {
  const chip = document.createElement("span");
  chip.textContent = value;
  return chip;
}

function reloadFrame() {
  const scenario = scenarios[selectedScenario];
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error("The target page did not load.")), 5000);
    ui.frame.addEventListener("load", () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
    ui.frame.src = `${scenario.url}${scenario.url.includes("?") ? "&" : "?"}run=${Date.now()}`;
  });
}

function selectScenario(name) {
  if (running || !scenarios[name]) return;
  selectedScenario = name;
  const scenario = scenarios[name];
  document.querySelectorAll(".scenario").forEach((button) => {
    button.classList.toggle("active", button.dataset.scenario === name);
  });
  ui.stage.className = `browser-stage ${scenario.stage}`;
  ui.address.textContent = scenario.address;
  ui.viewport.textContent = scenario.viewport;
  resetAudit();
  ui.frame.src = scenario.url;
}

function isRendered(element) {
  const style = element.ownerDocument.defaultView.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) !== 0 && rect.width > 0 && rect.height > 0;
}

function implicitRole(element) {
  const explicit = element.getAttribute("role");
  if (explicit) return normalize(explicit);
  const tag = element.tagName.toLowerCase();
  const type = normalize(element.getAttribute("type") || "text");
  if (tag === "button") return "button";
  if (tag === "select") return "combobox";
  if (tag === "textarea") return "textbox";
  if (tag === "input" && type === "checkbox") return "checkbox";
  if (tag === "input" && type !== "hidden") return "textbox";
  return "";
}

function accessibleName(element) {
  const aria = element.getAttribute("aria-label");
  if (aria) return normalize(aria);
  const id = element.id;
  if (id) {
    const label = [...element.ownerDocument.querySelectorAll("label")]
      .find((candidate) => candidate.htmlFor === id);
    if (label) return normalize(label.textContent);
  }
  const wrappingLabel = element.closest("label");
  if (wrappingLabel) return normalize(wrappingLabel.textContent);
  if (element.tagName.toLowerCase() === "button") return normalize(element.textContent);
  return normalize(element.getAttribute("placeholder") || element.getAttribute("name"));
}

function candidateRecord(element) {
  const attributes = {};
  for (const name of ["autocomplete", "name", "type", "aria-label", "placeholder", "data-action"]) {
    if (element.hasAttribute(name)) attributes[name] = normalize(element.getAttribute(name));
  }
  return {
    element,
    node_name: element.tagName.toLowerCase(),
    ax_role: implicitRole(element),
    ax_name: accessibleName(element),
    attributes,
  };
}

function resolveTarget(step) {
  const document = ui.frame.contentDocument;
  if (!document) throw new Error("The target frame is unavailable.");
  const all = [...document.querySelectorAll("input, select, textarea, button")]
    .filter(isRendered)
    .map(candidateRecord);
  return all.filter((candidate) => {
    const locator = step.locator;
    if (locator.node_name && candidate.node_name !== normalize(locator.node_name)) return false;
    if (locator.ax_role && candidate.ax_role !== normalize(locator.ax_role)) return false;
    if (locator.ax_name && candidate.ax_name !== normalize(locator.ax_name)) return false;
    return Object.entries(locator.attributes || {}).every(
      ([name, value]) => candidate.attributes[name] === normalize(value),
    );
  });
}

function locatorSummary(step) {
  const parts = [];
  if (step.locator.ax_role) parts.push(`role=${step.locator.ax_role}`);
  if (step.locator.ax_name) parts.push(`name=${step.locator.ax_name}`);
  for (const [name, value] of Object.entries(step.locator.attributes || {})) {
    parts.push(`${name}=${value}`);
  }
  return parts;
}

function updateResolver(step, matches) {
  ui.resolverTitle.textContent = step.label;
  ui.matchCount.textContent = `${matches.length} ${matches.length === 1 ? "match" : "matches"}`;
  ui.matchCount.className = `match-count ${matches.length === 1 ? "unique" : matches.length > 1 ? "ambiguous" : ""}`;
  ui.chips.replaceChildren(...locatorSummary(step).map(makeChip));
  ui.selector.textContent = locatorSummary(step).join(" · ");
  if (matches.length === 1) {
    const rect = matches[0].element.getBoundingClientRect();
    ui.geometry.textContent = `x ${Math.round(rect.x)} · y ${Math.round(rect.y)} · ${Math.round(rect.width)}×${Math.round(rect.height)}`;
  } else {
    ui.geometry.textContent = matches.length === 0 ? "no current target" : "discarded — ambiguous";
  }
}

function addTrace(step, result, detail) {
  ui.trace.querySelector(".empty-trace")?.remove();
  const item = document.createElement("li");
  item.className = `trace-item ${result}`;
  item.dataset.index = String(ui.trace.children.length + 1).padStart(2, "0");
  const title = document.createElement("strong");
  title.textContent = step.label;
  const explanation = document.createElement("p");
  explanation.textContent = detail;
  const badge = document.createElement("span");
  badge.className = "trace-result";
  badge.textContent = result === "refused" ? "needs recovery" : result;
  item.append(title, explanation, badge);
  ui.trace.append(item);
  item.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function showTarget(element) {
  ui.callout.hidden = false;
  ui.calloutText.textContent = "unique target · current DOM";
  element.scrollIntoView({ block: "center", behavior: "smooth" });
  const previousOutline = element.style.outline;
  const previousOffset = element.style.outlineOffset;
  element.style.outline = "3px solid #b8ff3e";
  element.style.outlineOffset = "3px";
  await wait(animationDelay);
  element.style.outline = previousOutline;
  element.style.outlineOffset = previousOffset;
  ui.callout.hidden = true;
}

function parameterValue(step, parameters) {
  return step.value?.parameter ? parameters[step.value.parameter] : step.value?.literal;
}

function executeAction(step, element, parameters) {
  const value = parameterValue(step, parameters);
  if (step.action === "input") {
    if (step.clear) element.value = "";
    element.value = value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return;
  }
  if (step.action === "select") {
    const option = [...element.options].find(
      (candidate) => normalize(candidate.value) === normalize(value) || normalize(candidate.textContent) === normalize(value),
    );
    if (!option) throw new Error(`The requested option is unavailable.`);
    element.value = option.value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
    return;
  }
  if (step.action === "click") element.click();
}

function postconditionMet(step, parameters) {
  const postcondition = step.postcondition;
  if (postcondition.type === "page_contains_text") {
    return normalize(ui.frame.contentDocument.body.innerText).includes(normalize(postcondition.text));
  }
  const matches = resolveTarget(step);
  if (postcondition.type === "target_disappears") return matches.length === 0;
  if (matches.length !== 1) return false;
  const element = matches[0].element;
  if (postcondition.type === "control_value_equals") {
    return normalize(element.value) === normalize(parameters[postcondition.parameter]);
  }
  if (postcondition.type === "checked") return element.checked === postcondition.value;
  return false;
}

async function verify(step, parameters, timeout = 3500) {
  const deadline = performance.now() + timeout;
  while (performance.now() < deadline) {
    if (postconditionMet(step, parameters)) return true;
    await wait(80);
  }
  return false;
}

function nextRequiredStep(index) {
  return procedure.steps.slice(index + 1).find((step) => !step.optional);
}

async function runReplay(event) {
  event.preventDefault();
  if (running || !procedure) return;
  setBusy(true);
  resetAudit();
  setStatus("running");
  ui.state.textContent = "Loading a fresh target page";
  const formData = new FormData(ui.form);
  const parameters = Object.fromEntries(formData.entries());
  try {
    await reloadFrame();
    for (let index = 0; index < procedure.steps.length; index += 1) {
      const step = procedure.steps[index];
      ui.state.textContent = `Resolving: ${step.label}`;
      const matches = resolveTarget(step);
      updateResolver(step, matches);
      await wait(animationDelay / 2);

      if (matches.length === 0 && step.optional) {
        const downstream = nextRequiredStep(index);
        const downstreamMatches = downstream ? resolveTarget(downstream) : [];
        if (downstreamMatches.length === 1) {
          addTrace(step, "skipped", `Optional route step absent; next required target is uniquely available. Coverage ${Math.round(step.route_coverage * 100)}%.`);
          continue;
        }
      }

      if (matches.length !== 1) {
        const reason = matches.length === 0
          ? "No exact semantic target exists in the current DOM. No action was taken."
          : `${matches.length} exact targets exist. Ranking is forbidden, so no action was taken.`;
        addTrace(step, "refused", reason);
        ui.state.textContent = "Stopped safely before action";
        setStatus("needs_recovery", "refused");
        return;
      }

      const element = matches[0].element;
      if (element.disabled) {
        addTrace(step, "refused", "The unique target is disabled. No action was taken.");
        ui.state.textContent = "Stopped safely before action";
        setStatus("needs_recovery", "refused");
        return;
      }

      await showTarget(element);
      executeAction(step, element, parameters);
      const verified = await verify(step, parameters);
      if (!verified) {
        addTrace(step, "refused", "The action ran, but the expected browser state was not observed before timeout.");
        ui.state.textContent = "Postcondition not proven";
        setStatus("needs_recovery", "refused");
        return;
      }
      addTrace(step, "executed", "Unique semantic match · action executed · postcondition verified.");
    }
    ui.state.textContent = "Registration verified by browser state";
    setStatus("completed");
  } catch (error) {
    const syntheticStep = { label: "Replay engine" };
    addTrace(syntheticStep, "refused", error instanceof Error ? error.message : "Unexpected local replay error.");
    ui.state.textContent = "Replay stopped";
    setStatus("needs_recovery", "refused");
  } finally {
    ui.callout.hidden = true;
    setBusy(false);
  }
}

document.querySelectorAll(".scenario").forEach((button) => {
  button.addEventListener("click", () => selectScenario(button.dataset.scenario));
});
ui.form.addEventListener("submit", runReplay);
ui.reset.addEventListener("click", () => selectScenario(selectedScenario));

fetch("demo-procedure.json")
  .then((response) => {
    if (!response.ok) throw new Error("The compiled procedure could not be loaded.");
    return response.json();
  })
  .then((loadedProcedure) => { procedure = loadedProcedure; })
  .catch((error) => {
    ui.state.textContent = error.message;
    setStatus("needs_recovery", "refused");
  });

resetAudit();
