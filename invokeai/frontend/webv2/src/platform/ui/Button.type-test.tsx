import { IconButton } from './Button';

const labelled = <IconButton aria-label="Open settings" />;
const labelledBy = <IconButton aria-labelledby="settings-label" />;

// @ts-expect-error Icon-only controls must expose an accessible name.
const unnamed = <IconButton />;

// @ts-expect-error Pick one accessible-name source so the contract stays unambiguous.
const ambiguouslyNamed = <IconButton aria-label="Open settings" aria-labelledby="settings-label" />;

void labelled;
void labelledBy;
void unnamed;
void ambiguouslyNamed;
