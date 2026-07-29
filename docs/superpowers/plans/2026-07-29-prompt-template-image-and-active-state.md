# Prompt Template Image and Active-State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore fetched prompt-template preview images under React StrictMode and render the applied template's selectable row with the shared solid accent treatment.

**Architecture:** Keep authenticated blob fetching in `PromptTemplateImage` and make the keyed `BlobImage` child own each object URL from mount setup through matching cleanup. Keep applied-template state in the existing panel contract and express its visual state through the shared `Row` recipe, leaving edit and delete actions as separate controls.

**Tech Stack:** React 19, TypeScript, Chakra UI v3, TanStack Query, Vitest Browser Mode, Playwright

## Global Constraints

- No backend, query-key, authentication, or prompt-template DTO changes.
- No new color tokens or row recipes.
- Local preview data URLs keep their current behavior.
- The previously added `border.image` semantic outline remains the single image-outline rule.
- Fetched blob URLs must be created during mount setup and the exact URL from each setup must be revoked by its matching cleanup.
- The selected template button must use `Row active="accent"`, expose `aria-current`, and use `accent.contrast` for both name and summary.
- Edit and delete controls must remain outside the accent surface.
- Do not call React `useEffect` directly; `useMountEffect` is the repository escape hatch for browser-resource setup and cleanup.

---

## File Structure

- `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplateImage.tsx` continues to select stored, local, removed, and fallback image states; its private keyed `BlobImage` child owns fetched object URLs.
- `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplatesPanel.tsx` continues to group and apply templates; `TemplateRow` composes the shared `Row` recipe around only the apply button.
- `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx` covers both user-visible regressions with the real Chakra system and TanStack Query provider.

### Task 1: Make fetched image object URLs StrictMode-safe

**Files:**
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplateImage.tsx:41-66`
- Test: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx:1-150,597-621`

**Interfaces:**
- Consumes: `PromptTemplateImageProps.fallback: ReactNode`, `promptTemplateImageQueryOptions(template.id)`, and the existing blob identity key from `getBlobKey(blob)`.
- Produces: private `BlobImage({ blob, fallback, imageProps }): ReactNode`, which renders the fallback before setup publishes a URL and an outlined `Image` afterward.

- [ ] **Step 1: Write the StrictMode regression test**

Import `StrictMode` from React. Replace the single-setup unmount test with a test whose `createObjectURL` spy returns a unique URL on every setup and whose `revokeObjectURL` spy records revoked URLs:

```tsx
it('keeps the fetched image URL live through StrictMode replay and revokes every owned URL', async () => {
  let sequence = 0;
  const revokedUrls = new Set<string>();
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:template-preview-${++sequence}`);
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation((url) => revokedUrls.add(url));

  await render(
    <StrictMode>
      <PromptTemplateImage alt="Fetched preview" fallback={IMAGE_FALLBACK} template={templateWithImage} />
    </StrictMode>,
    (queryClient) =>
      queryClient.setQueryData(promptTemplateKeys.image(templateWithImage.id), new Blob(['image']))
  );

  const image = host!.querySelector<HTMLImageElement>('img[alt="Fetched preview"]')!;
  const liveUrl = image.getAttribute('src')!;

  expect(sequence).toBeGreaterThan(1);
  expect(revokedUrls.has('blob:template-preview-1')).toBe(true);
  expect(revokedUrls.has(liveUrl)).toBe(false);
  expect(getComputedStyle(image).outlineWidth).toBe('1px');
  expect(getComputedStyle(image).outlineOffset).toBe('-1px');

  await act(() => root?.unmount());
  root = null;
  expect(revokedUrls.has(liveUrl)).toBe(true);
});
```

Add `vi.restoreAllMocks()` to the existing `afterEach` so a failed assertion cannot leak URL spies into later tests.

This test catches the current bug: moving URL creation back to a state initializer makes StrictMode revoke the URL still assigned to the rendered image.

- [ ] **Step 2: Run the focused browser test and verify RED**

Run from `invokeai/frontend/webv2`:

```bash
pnpm run test:browser -- src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
```

Expected: the StrictMode test fails because the rendered `src` is already present in `revokedUrls` and only one object URL was created.

- [ ] **Step 3: Move object URL ownership into the mount lifecycle**

Pass the existing fallback into `BlobImage`, initialize its URL state to `null`, and create/revoke each URL inside one `useMountEffect` setup:

```tsx
return (
  <BlobImage key={getBlobKey(query.data)} blob={query.data} fallback={fallback} imageProps={imageProps} />
);

const BlobImage = ({
  blob,
  fallback,
  imageProps,
}: {
  blob: Blob;
  fallback: ReactNode;
  imageProps: Omit<ImageProps, 'src'>;
}) => {
  const [src, setSrc] = useState<string | null>(null);

  useMountEffect(() => {
    const objectUrl = URL.createObjectURL(blob);
    setSrc(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  });

  return src ? <Image {...IMAGE_OUTLINE_PROPS} {...imageProps} src={src} /> : fallback;
};
```

Do not clear state from cleanup: final unmount does not need a render, and StrictMode's next setup immediately publishes its own URL.

- [ ] **Step 4: Run focused verification and verify GREEN**

Run from `invokeai/frontend/webv2`:

```bash
pnpm run test:browser -- src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
pnpm exec oxfmt --check src/features/generation/ui/promptFields/PromptTemplateImage.tsx src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
pnpm run lint:tsc
```

Expected: the focused browser file passes with no unhandled errors or React warnings, formatting passes, and TypeScript emits no errors.

- [ ] **Step 5: Commit the image lifecycle fix**

```bash
git add invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplateImage.tsx invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
git commit -m "fix(webv2): keep template image URLs live"
```

### Task 2: Highlight the applied template with the accent row recipe

**Files:**
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplatesPanel.tsx:1-14,352-418`
- Test: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx:1-180,183-220`

**Interfaces:**
- Consumes: `TemplateRow.isActive: boolean`, `RowProps.active: 'none' | 'muted' | 'brand' | 'accent'`, and `toPromptTemplateSnapshot(template)`.
- Produces: a native apply button composed with `Row asChild`, `active={isActive ? 'accent' : 'none'}`, and `aria-current={isActive || undefined}`.

- [ ] **Step 1: Add a stateful browser-test harness and failing active-row test**

Import `Box` from Chakra, `PromptTemplateSnapshot`, and React `useState`. Add this module-level test harness so clicking a real row updates the controlled `activeTemplate` prop:

```tsx
const StatefulPromptTemplatesPanel = ({ catalog }: { catalog: PromptTemplateCatalog }) => {
  const [activeTemplate, setActiveTemplate] = useState<PromptTemplateSnapshot | null>(null);

  return (
    <>
      <Box aria-hidden bg="accent.solid" color="accent.contrast" data-testid="accent-style-probe" />
      <PromptTemplatesPanel
        activeTemplate={activeTemplate}
        catalog={catalog}
        isActiveTemplateMissing={false}
        onApply={setActiveTemplate}
        onCreate={vi.fn()}
        onDetach={vi.fn()}
        onEdit={vi.fn()}
      />
    </>
  );
};
```

Add a panel test:

```tsx
it('marks the applied template as current with the accent surface and contrast text', async () => {
  await render(<StatefulPromptTemplatesPanel catalog={createCatalog()} />);

  const activeButton = buttonWithText('Cinematic');
  const inactiveButton = buttonWithText('Community');
  await act(async () => userEvent.click(activeButton));

  const probeStyle = getComputedStyle(host!.querySelector('[data-testid="accent-style-probe"]')!);
  const activeText = activeButton.querySelectorAll('span');

  expect(activeButton.getAttribute('aria-current')).toBe('true');
  expect(inactiveButton.hasAttribute('aria-current')).toBe(false);
  expect(getComputedStyle(activeButton).backgroundColor).toBe(probeStyle.backgroundColor);
  expect(getComputedStyle(inactiveButton).backgroundColor).not.toBe(probeStyle.backgroundColor);
  expect(activeText).toHaveLength(2);
  expect(getComputedStyle(activeText[0]!).color).toBe(probeStyle.color);
  expect(getComputedStyle(activeText[1]!).color).toBe(probeStyle.color);
});
```

The probe derives browser-resolved semantic-token colors independently of `TemplateRow`; the test catches removing `Row`, selecting the wrong active variant, omitting `aria-current`, or leaving active text on muted tokens.

- [ ] **Step 2: Run the focused browser test and verify RED**

Run from `invokeai/frontend/webv2`:

```bash
pnpm run test:browser -- src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
```

Expected: the new test fails because the applied button has no `aria-current` attribute and does not use the `accent.solid` background.

- [ ] **Step 3: Compose the apply button with the shared Row recipe**

Import `Row` from `@platform/ui/Row`. Replace only the selectable `Button` in `TemplateRow`:

```tsx
<Row
  active={isActive ? 'accent' : 'none'}
  aria-current={isActive || undefined}
  asChild
  flex="1"
  h="auto"
  justifyContent="start"
  minW="0"
  px="2"
  py="1.5"
  rounded="sm"
>
  <button type="button" onClick={handleApply}>
    <TemplateThumbnail template={template} />
    <Stack align="start" flex="1" gap="0" minW="0">
      <Text
        as="span"
        color={isActive ? 'accent.contrast' : 'fg.muted'}
        fontSize="xs"
        fontWeight={isActive ? '600' : '400'}
      >
        {template.name}
      </Text>
      <Text
        as="span"
        color={isActive ? 'accent.contrast' : 'fg.subtle'}
        fontFamily="mono"
        fontSize="2xs"
        truncate
      >
        {summary}
      </Text>
    </Stack>
  </button>
</Row>
```

Keep the surrounding `HStack`, edit button, and delete button unchanged so secondary and destructive actions are not part of the accent apply surface.

- [ ] **Step 4: Run focused and package verification and verify GREEN**

Run from `invokeai/frontend/webv2`:

```bash
pnpm run test:browser -- src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
pnpm exec oxfmt --check src/features/generation/ui/promptFields/PromptTemplatesPanel.tsx src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
pnpm run lint:oxc
pnpm run lint:tsc
```

Expected: the focused browser file passes with no unhandled errors or React warnings, formatting and Oxc lint pass without warnings, and TypeScript emits no errors.

- [ ] **Step 5: Commit the active row treatment**

```bash
git add invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplatesPanel.tsx invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
git commit -m "feat(webv2): highlight the applied prompt template"
```

## Final Verification

After both task reviews are clean, run from `invokeai/frontend/webv2`:

```bash
pnpm run format:check
pnpm run lint
pnpm run test:all
```

Expected: all formatting, Oxc, TypeScript, architecture, unit, browser, and fixture checks pass. Then review the full branch diff from the merge-base, explicitly triaging every deferred finding in the SDD ledger before declaring the PR ready.
