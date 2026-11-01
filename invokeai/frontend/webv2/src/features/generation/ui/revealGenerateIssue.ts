import type { GenerationIssue } from '@features/generation/core/generationIssues';

/** How long a reveal waits for the Generate panel, just opened, to render what it was asked to show. */
const REVEAL_WAIT_MS = 1500;

/**
 * Brings one Generate issue into view once the panel shows it: the note under its field (by `data-issue-id`), else its
 * section (by `data-generate-section`), for a control that shows the message itself. Focus goes to the note's
 * one-step fix, else the section header, so the next key acts where the problem is. The caller opens the panel and the
 * section first; a reveal whose panel never renders gives up quietly.
 */
export const revealGenerateIssue = (issue: Pick<GenerationIssue, 'id' | 'target'>): void => {
  const deadline = performance.now() + REVEAL_WAIT_MS;

  const attempt = () => {
    const frame = [...document.querySelectorAll<HTMLElement>('[data-hotkey-widget-type-id="generate"]')].find(
      (element) => element.getClientRects().length > 0
    );
    const note = frame?.querySelector<HTMLElement>(`[data-issue-id="${CSS.escape(issue.id)}"]`);
    const section = frame?.querySelector<HTMLElement>(`[data-generate-section="${CSS.escape(issue.target.section)}"]`);
    // A section the reveal just opened may not show its notes yet; it stands in only once it is open.
    const anchor = note ?? (section?.dataset.state === 'closed' ? undefined : section);

    if (!anchor) {
      if (performance.now() < deadline) {
        requestAnimationFrame(attempt);
      }
      return;
    }

    anchor.scrollIntoView({ block: 'nearest' });
    (
      note?.querySelector<HTMLElement>('[data-issue-fix]') ??
      section?.querySelector<HTMLElement>('[data-scope="collapsible"][data-part="trigger"], button, input')
    )?.focus({ preventScroll: true });
  };

  requestAnimationFrame(attempt);
};
