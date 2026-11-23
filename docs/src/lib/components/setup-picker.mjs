// @ts-check
// The "Your setup" picker: which operating system and GPU the install commands are shown for.
// Every variant is rendered at build time; the picker only toggles visibility, so readers
// without JavaScript and the search index still see every command.

/** @typedef {{ id: string, label: string }} SetupGpu */
/** @typedef {{ id: string, label: string, gpus: SetupGpu[] }} SetupOs */
/** @typedef {{ os: string, gpu: string }} Setup */
/**
 * A brand name as its `label`, or the UI string key of its label.
 * @typedef {{ id: string, label?: string, labelKey?: string }} SetupGpuOption
 */
/** @typedef {{ id: string, label: string, gpus: SetupGpuOption[] }} SetupOsOption */

export const SETUP_STORAGE_KEY = 'invoke-docs:setup';

/**
 * The first GPU of each OS is its default. Pages use these through `localizeSetupOptions`.
 * @type {SetupOsOption[]}
 */
export const SETUP_OPTIONS = [
  {
    id: 'windows',
    label: 'Windows',
    gpus: [
      { id: 'nvidia', label: 'NVIDIA' },
      { id: 'amd', label: 'AMD' },
      { id: 'intel', label: 'Intel Arc' },
      { id: 'none', labelKey: 'setupPicker.option.otherOrNone' },
      { id: 'arm64', label: 'ARM64' },
    ],
  },
  { id: 'macos', label: 'macOS', gpus: [{ id: 'apple', labelKey: 'setupPicker.option.appleSilicon' }] },
  {
    id: 'linux',
    label: 'Linux',
    gpus: [
      { id: 'nvidia', label: 'NVIDIA' },
      { id: 'amd', label: 'AMD' },
      { id: 'intel', label: 'Intel Arc' },
      { id: 'none', labelKey: 'setupPicker.option.noneCpu' },
    ],
  },
];

/**
 * The setup options with every label in the page's language.
 * @param {SetupOsOption[]} options
 * @param {(key: string) => string} t
 * @returns {SetupOs[]}
 */
export const localizeSetupOptions = (options, t) =>
  options.map((os) => ({
    ...os,
    gpus: os.gpus.map(({ id, label, labelKey }) => ({ id, label: labelKey ? t(labelKey) : String(label) })),
  }));

/**
 * Turns a stored (untrusted) value into a valid setup. A missing or unknown OS falls back to the
 * one detected from `platform`, then to the first option; a GPU the OS does not offer falls back to
 * that OS's default.
 *
 * Runs in the page's inline script as `resolveSetup.toString()`, so it must only use its arguments
 * and globals.
 *
 * @param {SetupOs[]} options
 * @param {unknown} candidate
 * @param {string | undefined} platform `navigator.userAgentData.platform` or `navigator.platform`
 * @returns {Setup}
 */
export function resolveSetup(options, candidate, platform) {
  const value = candidate && typeof candidate === 'object' ? /** @type {Partial<Setup>} */ (candidate) : {};
  const name = String(platform || '').toLowerCase();
  const detected = name.includes('mac') ? 'macos' : name.includes('linux') ? 'linux' : 'windows';
  const os =
    options.find((option) => option.id === value.os) ??
    options.find((option) => option.id === detected) ??
    options[0];
  const gpu = os.gpus.find((option) => option.id === value.gpu) ?? os.gpus[0];
  return { os: os.id, gpu: gpu.id };
}

/**
 * Validates the space-separated OS and GPU ids a variant applies to and builds its label.
 * Unknown ids throw, so a typo fails the build instead of hiding a command.
 *
 * @param {SetupOs[]} options
 * @param {{ os?: string, gpu?: string }} props
 * @returns {{ os?: string[], gpu?: string[], label: string }}
 */
export function describeVariant(options, { os, gpu }) {
  const osIds = os?.split(/\s+/).filter(Boolean);
  const gpuIds = gpu?.split(/\s+/).filter(Boolean);
  if (!osIds?.length && !gpuIds?.length) throw new Error('A setup variant needs an `os` or a `gpu`.');

  const scope = osIds?.length
    ? osIds.map((id) => {
        const match = options.find((option) => option.id === id);
        if (!match) throw new Error(`Unknown setup OS "${id}".`);
        return match;
      })
    : options;
  const gpuLabels = (gpuIds ?? []).map((id) => {
    const labels = scope.flatMap((option) => option.gpus.filter((g) => g.id === id).map((g) => g.label));
    if (!labels.length) throw new Error(`No setup OS in "${os ?? 'any'}" offers GPU "${id}".`);
    return labels[0];
  });

  const label = [osIds?.length ? scope.map((option) => option.label).join(', ') : '', gpuLabels.join(', ')]
    .filter(Boolean)
    .join(' · ');
  return { os: osIds?.length ? osIds : undefined, gpu: gpuIds?.length ? gpuIds : undefined, label };
}

/**
 * Lists the valid setups a set of alternatives misses or covers more than once. A `SetupSwitch`
 * must show exactly one alternative for every setup the picker offers.
 *
 * @param {SetupOs[]} options
 * @param {{ os?: string[], gpu?: string[] }[]} variants
 * @returns {{ missing: string[], repeated: string[] }}
 */
export function findCoverageGaps(options, variants) {
  /** @type {{ missing: string[], repeated: string[] }} */
  const gaps = { missing: [], repeated: [] };
  for (const os of options) {
    for (const gpu of os.gpus) {
      const matches = variants.filter(
        (variant) => (!variant.os || variant.os.includes(os.id)) && (!variant.gpu || variant.gpu.includes(gpu.id))
      ).length;
      const label = `${os.label} · ${gpu.label}`;
      if (matches === 0) gaps.missing.push(label);
      if (matches > 1) gaps.repeated.push(label);
    }
  }
  return gaps;
}

/**
 * Reads the OS and GPU ids of the `SetupVariant` elements in rendered HTML.
 *
 * @param {string} html
 * @returns {{ os?: string[], gpu?: string[] }[]}
 */
export function variantsInHtml(html) {
  return [...html.matchAll(/<div\b[^>]*\bdata-setup-variant\b[^>]*>/g)].map(([tag]) => {
    const ids = (/** @type {string} */ name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1].split(' ');
    return { os: ids('data-setup-for-os'), gpu: ids('data-setup-for-gpu') };
  });
}

/**
 * Hides every element marked for another OS or GPU once a setup is chosen. Without JavaScript the
 * root never gets `data-setup-os`, so nothing is hidden.
 *
 * @param {SetupOs[]} options
 */
export function setupVisibilityCss(options) {
  const osIds = options.map((option) => option.id);
  const gpuIds = [...new Set(options.flatMap((option) => option.gpus.map((gpu) => gpu.id)))];
  return [
    ...osIds.map(
      (id) => `:root[data-setup-os="${id}"] [data-setup-for-os]:not([data-setup-for-os~="${id}"])`
    ),
    ...gpuIds.map(
      (id) => `:root[data-setup-gpu="${id}"] [data-setup-for-gpu]:not([data-setup-for-gpu~="${id}"])`
    ),
  ]
    .map((selector) => `${selector}{display:none}`)
    .join('\n');
}

/**
 * Applies the stored or detected setup and wires the picker's radios. Runs as an inline script at
 * the end of the picker, before the commands below it are parsed, so the page never paints the
 * wrong variant. Serialized with `toString()`: it must only use its arguments and globals.
 *
 * @param {HTMLElement} picker
 * @param {SetupOs[]} options
 * @param {typeof resolveSetup} resolve
 * @param {string} storageKey
 */
export function mountSetupPicker(picker, options, resolve, storageKey) {
  let stored = null;
  try {
    stored = JSON.parse(localStorage.getItem(storageKey) || 'null');
  } catch {
    // Storage blocked or holding junk: fall back to detection.
  }
  const nav = /** @type {Navigator & { userAgentData?: { platform?: string } }} */ (navigator);
  const platform = nav.userAgentData?.platform || nav.platform;
  const summary = picker.querySelector('[data-setup-summary]');
  /** @type {NodeListOf<HTMLInputElement>} */
  const inputs = picker.querySelectorAll('input[type="radio"]');

  /** @param {Setup} setup */
  const apply = (setup) => {
    const os = options.find((option) => option.id === setup.os) ?? options[0];
    const gpu = os.gpus.find((option) => option.id === setup.gpu) ?? os.gpus[0];
    document.documentElement.dataset.setupOs = os.id;
    document.documentElement.dataset.setupGpu = gpu.id;
    for (const input of inputs) {
      if (input.name === 'setup-gpu') input.disabled = !os.gpus.some((option) => option.id === input.value);
      input.checked = input.value === (input.name === 'setup-os' ? os.id : gpu.id);
    }
    if (summary) summary.textContent = `${os.label} · ${gpu.label}`;
  };

  let current = resolve(options, stored, platform);
  apply(current);
  picker.addEventListener('change', (event) => {
    const input = /** @type {HTMLInputElement} */ (event.target);
    const next = input.name === 'setup-os' ? { os: input.value, gpu: current.gpu } : { os: current.os, gpu: input.value };
    current = resolve(options, next, platform);
    apply(current);
    try {
      localStorage.setItem(storageKey, JSON.stringify(current));
    } catch {
      // Storage blocked: the choice still applies to this page.
    }
  });
}
