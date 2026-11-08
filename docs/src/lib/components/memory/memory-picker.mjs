// @ts-check
// The "Which memory option do I need?" picker (`MemoryPicker.astro`): from the reader's setup (the shared OS and GPU
// picker, `setup-picker.mjs`), NVIDIA generation, VRAM and model format, which of the page's options apply. Every
// option is rendered at build time, in the page's order; the picker only hides the others, so readers without
// JavaScript and the search index see every option.
//
// The rules follow the code:
// - Partial loading runs on CUDA (NVIDIA, AMD through ROCm) and Intel XPU; CPU and Apple MPS load models whole
//   (`ModelCache`, invokeai/backend/model_manager/load/model_cache/model_cache.py).
// - FP8 Storage is used on CUDA and XPU devices that pass its probe, never on CPU or MPS
//   (`_device_supports_fp8_storage`, invokeai/backend/model_manager/load/load_default.py).
// - fp8_compute needs compute capability 8.9 (RTX 40) or newer on NVIDIA, or a ROCm GPU that passes an fp8 matmul
//   probe; never XPU (invokeai/backend/quantization/fp8_scaled.py).
// - int8_compute needs NVIDIA compute capability 8.0 (RTX 30) or newer and Triton
//   (invokeai/backend/quantization/int8_matmul.py).
// - NF4 needs bitsandbytes, which the install leaves out on macOS (pyproject.toml); its loader has no device check, and
//   NF4 on Intel GPUs or the CPU is untested, so the picker does not present it as working there.
// - Windows on ARM64 means an NVIDIA RTX Spark: a CUDA device. Its install has no Triton (pyproject.toml), so
//   int8_compute falls back there; fp8_compute is left to its own probe.

/** @typedef {{ os: string, gpu: string, series: string, vram: string, format: string }} MemoryChoice */
/** @typedef {'series' | 'vram' | 'format'} MemoryRow */
/** @typedef {{ id: string, label?: string, labelKey?: string }} ChoiceOption */
/**
 * An option applies when any of its conditions holds; a condition holds when the choice's value for each key it
 * names is in that key's list.
 * @typedef {{ id: string, when: Partial<Record<keyof MemoryChoice, string[]>>[] }} MemoryRule
 */

export const MEMORY_STORAGE_KEY = 'invoke-docs:memory-picker';

/**
 * The rows below the setup picker, in the order their chips show, and the choice each starts on.
 * @type {Record<MemoryRow, { options: ChoiceOption[], initial: string }>}
 */
export const MEMORY_ROWS = {
  series: {
    initial: 'ada',
    options: [
      { id: 'pre-ampere', labelKey: 'memoryPicker.series.preAmpere' },
      { id: 'ampere', labelKey: 'memoryPicker.series.ampere' },
      { id: 'ada', labelKey: 'memoryPicker.series.ada' },
    ],
  },
  vram: {
    initial: '8',
    options: [
      { id: '6', labelKey: 'memoryPicker.vram.6' },
      { id: '8', labelKey: 'memoryPicker.vram.8' },
      { id: '12', labelKey: 'memoryPicker.vram.12' },
      { id: '16', labelKey: 'memoryPicker.vram.16' },
      { id: '24', labelKey: 'memoryPicker.vram.24' },
    ],
  },
  format: {
    initial: 'full',
    options: [
      { id: 'full', labelKey: 'memoryPicker.format.full' },
      { id: 'fp8', label: 'fp8 / fp8 scaled' },
      { id: 'mxfp8', label: 'MXFP8' },
      { id: 'int8', label: 'int8' },
      { id: 'gguf', label: 'GGUF' },
      { id: 'sdnq', label: 'SDNQ' },
      { id: 'nf4', label: 'NF4' },
      { id: 'nvfp4', label: 'nvfp4' },
    ],
  },
};

/** GPUs whose models live in system memory: no VRAM to budget, no partial loading. */
const SYSTEM_MEMORY = ['apple', 'none'];
/** GPUs with VRAM of their own that Invoke manages: CUDA (NVIDIA, AMD ROCm, RTX Spark) and Intel XPU. */
const GPUS = ['nvidia', 'amd', 'intel', 'arm64'];

/**
 * Every option the page renders. The page decides their order.
 * @type {MemoryRule[]}
 */
export const MEMORY_RULES = [
  { id: 'system-memory', when: [{ gpu: SYSTEM_MEMORY }] },
  { id: 'nf4-support', when: [{ gpu: ['apple', 'intel', 'none'], format: ['nf4'] }] },
  { id: 'sysmem-fallback', when: [{ os: ['windows'], gpu: ['nvidia'] }] },
  { id: 'rocm-windows', when: [{ os: ['windows'], gpu: ['amd'] }] },
  { id: 'low-vram-mode', when: [{ gpu: GPUS }] },
  { id: 'fp8-storage', when: [{ gpu: GPUS, format: ['full'] }] },
  { id: 'fp8-storage-installed', when: [{ gpu: GPUS, format: ['fp8'] }] },
  {
    id: 'fp8-compute',
    when: [
      { gpu: ['nvidia'], series: ['ada'], format: ['fp8'] },
      { gpu: ['amd', 'arm64'], format: ['fp8'] },
    ],
  },
  { id: 'mxfp8-off', when: [{ gpu: GPUS, format: ['mxfp8'] }] },
  { id: 'int8-compute', when: [{ gpu: ['nvidia'], series: ['ampere', 'ada'], format: ['int8'] }] },
  {
    id: 'int8-unpacked',
    when: [
      { gpu: ['amd', 'intel', 'arm64', 'apple', 'none'], format: ['int8'] },
      { gpu: ['nvidia'], series: ['pre-ampere'], format: ['int8'] },
    ],
  },
  {
    id: 'already-quantized',
    when: [{ format: ['gguf', 'sdnq', 'nvfp4'] }, { gpu: ['nvidia', 'amd', 'arm64'], format: ['nf4'] }],
  },
  {
    id: 'quantized-build',
    when: [
      { gpu: SYSTEM_MEMORY, format: ['full'] },
      { gpu: GPUS, vram: ['6', '8', '12'], format: ['full'] },
    ],
  },
  { id: 'tiled-decode', when: [{ gpu: GPUS, vram: ['6', '8', '12', '16'] }] },
  { id: 'fits-in-vram', when: [{ gpu: GPUS, vram: ['24'] }] },
  { id: 'still-oom', when: [{ gpu: GPUS }] },
  { id: 'page-file', when: [{ os: ['windows'] }] },
];

/**
 * Which rows matter for a GPU: the NVIDIA generation only for NVIDIA, VRAM only where the GPU has its own.
 * @type {{ row: MemoryRow, gpus: string[] }[]}
 */
export const ROW_VISIBILITY = [
  { row: 'series', gpus: ['nvidia'] },
  { row: 'vram', gpus: GPUS },
];

/**
 * The ids of the options that apply. Runs in the page's inline script as `toString()`, so
 * it must only use its arguments.
 * @param {MemoryRule[]} rules
 * @param {MemoryChoice} choice
 * @returns {string[]}
 */
export function recommend(rules, choice) {
  return rules
    .filter((rule) =>
      rule.when.some((condition) =>
        Object.entries(condition).every(([key, values]) =>
          values.includes(choice[/** @type {keyof MemoryChoice} */ (key)])
        )
      )
    )
    .map((rule) => rule.id);
}

/**
 * Turns stored (untrusted) row values into valid ones; anything unknown takes the row's initial choice. Runs in the
 * page's inline script as `toString()`, so it must only use its arguments.
 * @param {typeof MEMORY_ROWS} rows
 * @param {unknown} candidate
 * @returns {Record<MemoryRow, string>}
 */
export function resolveMemoryRows(rows, candidate) {
  const value = candidate && typeof candidate === 'object' ? /** @type {Record<string, unknown>} */ (candidate) : {};
  /** @type {Record<string, string>} */
  const resolved = {};
  for (const [row, { options, initial }] of Object.entries(rows)) {
    resolved[row] = options.some((option) => option.id === value[row]) ? String(value[row]) : initial;
  }
  return /** @type {Record<MemoryRow, string>} */ (resolved);
}

/**
 * Compares the options a page renders with the rules, so a typo or a forgotten option fails the build instead of
 * leaving a recommendation that never shows.
 * @param {string} html the picker's rendered options
 * @returns {{ missing: string[], unknown: string[] }}
 */
export function checkOptions(html) {
  const ids = [...html.matchAll(/\bdata-memory-option="([^"]*)"/g)].map(([, id]) => id);
  const known = MEMORY_RULES.map((rule) => rule.id);
  return {
    missing: known.filter((id) => !ids.includes(id)),
    unknown: ids.filter((id, index) => !known.includes(id) || ids.indexOf(id) !== index),
  };
}

/**
 * A row's chips with their labels in the page's language.
 * @param {ChoiceOption[]} options
 * @param {(key: string) => string} t
 */
export const localizeChoices = (options, t) =>
  options.map(({ id, label, labelKey }) => ({ id, label: labelKey ? t(labelKey) : String(label) }));

/**
 * Applies the stored rows and the shared setup, and re-applies them on every change. Runs as an inline script right
 * after the options, before they paint, so the page never shows options that do not apply. Serialized with
 * `toString()`: it must only use its arguments and globals.
 *
 * @param {HTMLElement} root the picker: the setup picker's radios and the rows' radios
 * @param {HTMLElement} list the options' container
 * @param {{ rows: typeof MEMORY_ROWS, rules: MemoryRule[], visibility: typeof ROW_VISIBILITY, storageKey: string,
 *   countForms: Record<string, string>, lang: string }} config
 * @param {typeof recommend} recommendFn
 * @param {typeof resolveMemoryRows} resolveFn
 */
export function mountMemoryPicker(root, list, config, recommendFn, resolveFn) {
  let stored = null;
  try {
    stored = JSON.parse(localStorage.getItem(config.storageKey) || 'null');
  } catch {
    // Storage blocked or holding junk: start from the initial choices.
  }
  let rows = resolveFn(config.rows, stored);
  /** @type {NodeListOf<HTMLElement>} */
  const options = list.querySelectorAll('[data-memory-option]');
  const summary = root.querySelector('[data-memory-summary]');

  const apply = () => {
    const html = document.documentElement;
    const gpu = html.dataset.setupGpu || '';
    const choice = { os: html.dataset.setupOs || '', gpu, ...rows };
    const ids = recommendFn(config.rules, choice);
    for (const option of options) option.hidden = !ids.includes(option.dataset.memoryOption || '');
    for (const { row, gpus } of config.visibility) {
      const fieldset = root.querySelector(`[data-memory-row="${row}"]`);
      if (fieldset instanceof HTMLElement) fieldset.hidden = !gpus.includes(gpu);
    }
    for (const input of root.querySelectorAll('input[data-memory-input]')) {
      const radio = /** @type {HTMLInputElement} */ (input);
      radio.checked = rows[/** @type {MemoryRow} */ (radio.name.replace('memory-', ''))] === radio.value;
    }
    if (summary) {
      const count = ids.length;
      const form = config.countForms[new Intl.PluralRules(config.lang).select(count)] ?? config.countForms.other ?? '';
      summary.textContent = form.replace(/\{\{\s*count\s*\}\}/g, String(count));
    }
  };

  apply();
  root.dataset.memoryReady = '';
  // The setup picker's own listener sits on an element inside the root, so it has updated the setup by now.
  root.addEventListener('change', (event) => {
    const input = /** @type {HTMLInputElement} */ (event.target);
    if (input.dataset.memoryInput !== undefined) {
      rows = resolveFn(config.rows, { ...rows, [input.name.replace('memory-', '')]: input.value });
      try {
        localStorage.setItem(config.storageKey, JSON.stringify(rows));
      } catch {
        // Storage blocked: the choice still applies to this page.
      }
    }
    apply();
  });
}
