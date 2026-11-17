import type { ReactNode } from 'react';

import { createKeyedTransientStore } from '@platform/state/externalStore';
import { createContext, use, useId } from 'react';

/** A mounted section, as the panel's group actions, title keys and jump menu reach it. */
export interface SectionPanelEntry {
  /** Stable for the section's mount; keys the jump menu's items. */
  key: string;
  label: string;
  /** The `.generate-section` root; its `data-state` is the section's open state. */
  root: HTMLElement;
  trigger: HTMLElement;
  /** Persisted sections write through the preference port, so a panel-wide change is one write. */
  sectionId: string | undefined;
  /** For a section without `sectionId`, whose open state is its own. */
  setLocalOpen: (open: boolean) => void;
}

type PersistSectionsOpen = (changes: Readonly<Record<string, boolean>>) => void;

const NO_ENTRIES: readonly SectionPanelEntry[] = [];

/**
 * Sections by panel id. A widget's header and body render apart, so the header's jump menu finds the body's sections
 * here under the widget instance id. Entries come and go with their sections' mounts.
 */
const panels = createKeyedTransientStore<string, readonly SectionPanelEntry[]>();

const byDocumentOrder = (left: SectionPanelEntry, right: SectionPanelEntry) =>
  left.root.compareDocumentPosition(right.root) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;

/** Adds a mounted section to its panel, in document order; the returned cleanup removes it. */
export const registerPanelSection = (panelId: string, entry: SectionPanelEntry): (() => void) => {
  panels.set(panelId, [...(panels.get(panelId) ?? NO_ENTRIES), entry].sort(byDocumentOrder));

  return () => {
    const remaining = (panels.get(panelId) ?? NO_ENTRIES).filter((candidate) => candidate !== entry);

    if (remaining.length === 0) {
      panels.delete(panelId);
    } else {
      panels.set(panelId, remaining);
    }
  };
};

export const getPanelSections = (panelId: string): readonly SectionPanelEntry[] => panels.get(panelId) ?? NO_ENTRIES;

/** The panel's sections in document order, for a view that lists them. */
export const usePanelSections = (panelId: string | undefined): readonly SectionPanelEntry[] =>
  panels.useValue(panelId ?? '') ?? NO_ENTRIES;

export const isPanelSectionOpen = (entry: SectionPanelEntry): boolean => entry.root.dataset.state === 'open';

/** Applies several open states at once: one preference write for the persisted sections, local state for the rest. */
export const setPanelSectionsOpen = (
  changes: ReadonlyArray<readonly [SectionPanelEntry, boolean]>,
  persist: PersistSectionsOpen
): void => {
  const persisted: Record<string, boolean> = {};

  for (const [entry, open] of changes) {
    if (isPanelSectionOpen(entry) === open) {
      continue;
    }
    if (entry.sectionId === undefined) {
      entry.setLocalOpen(open);
    } else {
      persisted[entry.sectionId] = open;
    }
  }

  if (Object.keys(persisted).length > 0) {
    persist(persisted);
  }
};

/** Opens `target` and folds every other section in its panel. */
export const soloPanelSection = (
  entries: readonly SectionPanelEntry[],
  target: HTMLElement,
  persist: PersistSectionsOpen
): void =>
  setPanelSectionsOpen(
    entries.map((entry) => [entry, entry.root === target] as const),
    persist
  );

/** Opens every section when any is folded; folds them all otherwise. */
export const toggleAllPanelSections = (entries: readonly SectionPanelEntry[], persist: PersistSectionsOpen): void => {
  const open = entries.some((entry) => !isPanelSectionOpen(entry));

  setPanelSectionsOpen(
    entries.map((entry) => [entry, open] as const),
    persist
  );
};

/** The settle-scroll still waiting on each section's opening, so a newer reveal replaces it. */
const pendingSettles = new WeakMap<HTMLElement, AbortController>();
/** Past any opening animation (reduced motion included); a settle not heard by then is dropped. */
const SETTLE_TIMEOUT_MS = 1000;

/**
 * Brings a section to the top of its panel with its title focused, opening it first when it is folded. The scroll
 * repeats once the opening animation ends: until then the panel may be too short to put the section at the top. A
 * fold, unmount or newer reveal before then drops the repeat.
 */
export const revealPanelSection = (entry: SectionPanelEntry, persist: PersistSectionsOpen): void => {
  const wasOpen = isPanelSectionOpen(entry);

  pendingSettles.get(entry.root)?.abort();
  if (!wasOpen) {
    setPanelSectionsOpen([[entry, true]], persist);
  }

  // After the menu that asked for this has handed focus back to its trigger.
  requestAnimationFrame(() => {
    entry.root.scrollIntoView({ block: 'start' });
    entry.trigger.focus({ preventScroll: true });

    const content = wasOpen ? null : entry.root.querySelector(':scope > [data-part="content"]');

    if (!content) {
      return;
    }

    const settle = new AbortController();
    const timeout = window.setTimeout(() => settle.abort(), SETTLE_TIMEOUT_MS);

    pendingSettles.set(entry.root, settle);
    settle.signal.addEventListener('abort', () => window.clearTimeout(timeout));
    content.addEventListener(
      'animationend',
      (event) => {
        // Animations inside the fields bubble here too; only the content's own ends the opening.
        if (event.target !== content) {
          return;
        }

        settle.abort();
        if (entry.root.isConnected && isPanelSectionOpen(entry)) {
          entry.root.scrollIntoView({ block: 'start' });
        }
      },
      { signal: settle.signal }
    );
  });
};

const SectionPanelContext = createContext<string | null>(null);

/** The panel id the sections inside belong to, or null outside a panel. */
export const useSectionPanelId = (): string | null => use(SectionPanelContext);

/**
 * Groups the sections inside into one panel: solo and expand-all act on them together, and title arrow keys move
 * between them. Pass the widget instance id so its header's jump menu finds the same sections.
 */
export const GenerationSectionPanel = ({ children, id }: { children: ReactNode; id?: string }) => {
  const fallbackId = useId();

  return <SectionPanelContext value={id ?? fallbackId}>{children}</SectionPanelContext>;
};
