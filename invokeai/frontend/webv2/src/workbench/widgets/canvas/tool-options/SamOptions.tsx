import type { SelectObjectSaveTarget } from '@workbench/canvas-engine/engine';
import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { ChangeEvent } from 'react';

import { HStack, Input, Menu, NativeSelect, Portal, Switch, Text } from '@chakra-ui/react';
import { Button, MenuContent } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import { isSamDocumentInputValid } from '@workbench/generation/canvas/samGraph';
import { useSamSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

export interface SamActionEligibility {
  canApply: boolean;
  canCancel: boolean;
  canProcess: boolean;
  canReset: boolean;
  canSave: boolean;
}

export const getSamActionEligibility = (session: SamSessionSnapshot): SamActionEligibility => {
  const isProcessing = session.status === 'processing';
  const hasReadyPreview = session.hasPreview && !isProcessing;
  return {
    canApply: hasReadyPreview,
    canCancel: true,
    canProcess: !isProcessing && isSamDocumentInputValid(session.input),
    canReset: true,
    canSave: hasReadyPreview,
  };
};

const SamSwitch = ({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange(checked: boolean): void;
}) => {
  const handleChange = useCallback(({ checked }: { checked: boolean }) => onChange(checked), [onChange]);
  return (
    <Switch.Root checked={checked} disabled={disabled} size="sm" onCheckedChange={handleChange}>
      <Switch.HiddenInput />
      <Switch.Control>
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="2xs">{label}</Switch.Label>
    </Switch.Root>
  );
};

const SAVE_TARGETS: readonly SelectObjectSaveTarget[] = ['raster', 'control', 'inpaint_mask', 'regional_guidance'];

const SamSaveItem = ({ engine, target }: ToolOptionsComponentProps & { target: SelectObjectSaveTarget }) => {
  const { t } = useTranslation();
  const save = useCallback(() => void engine.saveSelectObjectSession(target, makeImageDurable), [engine, target]);
  return (
    <Menu.Item value={target} onClick={save}>
      <Menu.ItemText>{t(`widgets.layers.selectObject.saveAs_${target}`)}</Menu.ItemText>
    </Menu.Item>
  );
};

export const SamOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const session = useSamSession(engine);

  const setBoolean = useCallback(
    (key: 'applyPolygonRefinement' | 'autoProcess' | 'invert' | 'isolatedPreview', value: boolean) =>
      engine.updateSelectObjectSession({ [key]: value }),
    [engine]
  );
  const setVisualMode = useCallback(
    () =>
      engine.updateSelectObjectSession({ input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' } }),
    [engine]
  );
  const setPromptMode = useCallback(
    () => engine.updateSelectObjectSession({ input: { prompt: '', type: 'prompt' } }),
    [engine]
  );
  const setInclude = useCallback(() => engine.updateSelectObjectSession({ pointLabel: 'include' }), [engine]);
  const setExclude = useCallback(() => engine.updateSelectObjectSession({ pointLabel: 'exclude' }), [engine]);
  const handlePrompt = useCallback(
    (event: ChangeEvent<HTMLInputElement>) =>
      engine.updateSelectObjectSession({ input: { prompt: event.currentTarget.value, type: 'prompt' } }),
    [engine]
  );
  const handleModel = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) =>
      engine.updateSelectObjectSession({ model: event.currentTarget.value as SamModel }),
    [engine]
  );
  const process = useCallback(() => void engine.processSelectObjectSession(), [engine]);
  const reset = useCallback(() => engine.resetSelectObjectSession(), [engine]);
  const apply = useCallback(() => void engine.applySelectObjectSession(), [engine]);
  const cancel = useCallback(() => engine.cancelSelectObjectSession(), [engine]);
  const setAutoProcess = useCallback((value: boolean) => setBoolean('autoProcess', value), [setBoolean]);
  const setIsolatedPreview = useCallback((value: boolean) => setBoolean('isolatedPreview', value), [setBoolean]);
  const setInvert = useCallback((value: boolean) => setBoolean('invert', value), [setBoolean]);
  const setRefinement = useCallback((value: boolean) => setBoolean('applyPolygonRefinement', value), [setBoolean]);

  if (!session) {
    return null;
  }

  const eligibility = getSamActionEligibility(session);
  const isProcessing = session.status === 'processing';

  return (
    <HStack align="center" gap="2" maxW="calc(100vw - 2rem)" overflowX="auto">
      <SamSwitch
        checked={session.autoProcess}
        label={t('widgets.layers.selectObject.autoProcess')}
        onChange={setAutoProcess}
      />
      <SamSwitch
        checked={session.isolatedPreview}
        label={t('widgets.layers.selectObject.isolatedPreview')}
        onChange={setIsolatedPreview}
      />
      <HStack gap="1" role="tablist">
        <Button
          aria-selected={session.input.type === 'visual'}
          role="tab"
          size="xs"
          variant={session.input.type === 'visual' ? 'solid' : 'ghost'}
          onClick={setVisualMode}
        >
          {t('widgets.layers.selectObject.visual')}
        </Button>
        <Button
          aria-selected={session.input.type === 'prompt'}
          role="tab"
          size="xs"
          variant={session.input.type === 'prompt' ? 'solid' : 'ghost'}
          onClick={setPromptMode}
        >
          {t('widgets.layers.selectObject.prompt')}
        </Button>
      </HStack>
      <SamSwitch checked={session.invert} label={t('widgets.layers.selectObject.invert')} onChange={setInvert} />
      {session.input.type === 'visual' ? (
        <HStack gap="1">
          <Button
            aria-pressed={session.pointLabel === 'include'}
            size="xs"
            variant={session.pointLabel === 'include' ? 'solid' : 'ghost'}
            onClick={setInclude}
          >
            {t('widgets.layers.selectObject.include')}
          </Button>
          <Button
            aria-pressed={session.pointLabel === 'exclude'}
            size="xs"
            variant={session.pointLabel === 'exclude' ? 'solid' : 'ghost'}
            onClick={setExclude}
          >
            {t('widgets.layers.selectObject.exclude')}
          </Button>
        </HStack>
      ) : (
        <Input
          aria-label={t('widgets.layers.selectObject.prompt')}
          autoComplete="off"
          size="xs"
          value={session.input.prompt}
          w="12rem"
          onChange={handlePrompt}
        />
      )}
      <NativeSelect.Root disabled={isProcessing} size="xs" w="10rem">
        <NativeSelect.Field
          aria-label={t('widgets.layers.selectObject.model')}
          value={session.model}
          onChange={handleModel}
        >
          <option value="segment-anything-2-large">{t('widgets.layers.selectObject.modelSam2Large')}</option>
          <option value="segment-anything-huge">{t('widgets.layers.selectObject.modelHuge')}</option>
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
      <SamSwitch
        checked={session.applyPolygonRefinement}
        disabled={isProcessing}
        label={t('widgets.layers.selectObject.refine')}
        onChange={setRefinement}
      />
      <Button disabled={!eligibility.canProcess} loading={isProcessing} size="xs" onClick={process}>
        {t('widgets.layers.selectObject.process')}
      </Button>
      <Button disabled={!eligibility.canReset} size="xs" variant="ghost" onClick={reset}>
        {t('widgets.layers.selectObject.reset')}
      </Button>
      <Button disabled={!eligibility.canApply} size="xs" onClick={apply}>
        {t('common.apply')}
      </Button>
      <Menu.Root>
        <Menu.Trigger asChild>
          <Button disabled={!eligibility.canSave} size="xs" variant="ghost">
            {t('widgets.layers.selectObject.saveAs')} <ChevronDownIcon size={12} />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="11rem" py="1">
              {SAVE_TARGETS.map((target) => (
                <SamSaveItem key={target} engine={engine} target={target} />
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Button disabled={!eligibility.canCancel} size="xs" variant="ghost" onClick={cancel}>
        {t('common.cancel')}
      </Button>
      {session.error ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {session.error}
        </Text>
      ) : null}
    </HStack>
  );
};
