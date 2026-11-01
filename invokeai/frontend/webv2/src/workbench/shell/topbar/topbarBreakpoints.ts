const hideBelow = (px: number) => ({ [`@media (max-width: ${px - 1}px)`]: { display: 'none' } }) as const;

const showBelow = (px: number) => ({ [`@media (min-width: ${px}px)`]: { display: 'none' } }) as const;

const PROJECT_NAME_MIN_WIDTH = 1024;

export const HIDE_BELOW_HINT_WIDTH = hideBelow(1440);

export const HIDE_BELOW_PRESET_LABEL_WIDTH = hideBelow(1280);

export const HIDE_BELOW_PROJECT_NAME_WIDTH = hideBelow(PROJECT_NAME_MIN_WIDTH);

/** Stands in for the project name where `HIDE_BELOW_PROJECT_NAME_WIDTH` hides it. */
export const SHOW_BELOW_PROJECT_NAME_WIDTH = showBelow(PROJECT_NAME_MIN_WIDTH);
