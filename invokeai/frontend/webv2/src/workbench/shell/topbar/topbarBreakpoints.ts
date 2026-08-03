const hideBelow = (px: number) => ({ [`@media (max-width: ${px - 1}px)`]: { display: 'none' } }) as const;

export const HIDE_BELOW_HINT_WIDTH = hideBelow(1440);

export const HIDE_BELOW_PRESET_LABEL_WIDTH = hideBelow(1280);

export const HIDE_BELOW_PROJECT_NAME_WIDTH = hideBelow(1024);
