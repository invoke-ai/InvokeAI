import { Flex, IconButton, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsEnabled } from 'features/controlLayers/hooks/useEntityIsEnabled';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import {
  getControlLayerWarnings,
  getGlobalReferenceImageWarnings,
  getInpaintMaskWarnings,
  getRasterLayerWarnings,
  getRegionalGuidanceWarnings,
} from 'features/controlLayers/store/validators';
import type { TFunction } from 'i18next';
import { upperFirst } from 'lodash-es';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const buildSelectWarnings = (entityIdentifier: CanvasEntityIdentifier, t: TFunction) => {
  return createSelector(selectCanvasSlice, selectModel, (canvas, model) => {
    // This component is used within a <CanvasEntityStateGate /> so we can safely assume that the entity exists.
    // Should never throw.
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'CanvasEntityHeaderWarnings');

    let warnings: string[] = [];

    const entityType = entity.type;

    if (entityType === 'control_layer') {
      warnings = getControlLayerWarnings(entity, model);
    } else if (entityType === 'regional_guidance') {
      warnings = getRegionalGuidanceWarnings(entity, model);
    } else if (entityType === 'inpaint_mask') {
      warnings = getInpaintMaskWarnings(entity, model);
    } else if (entityType === 'raster_layer') {
      warnings = getRasterLayerWarnings(entity, model);
    } else if (entityType === 'reference_image') {
      warnings = getGlobalReferenceImageWarnings(entity, model);
    } else {
      assert<Equals<typeof entityType, never>>(false, 'Unexpected entity type');
    }

    // Return a stable reference if there are no warnings
    if (warnings.length === 0) {
      return EMPTY_ARRAY;
    }

    return warnings.map((w) => t(w)).map(upperFirst);
  });
};

export const CanvasEntityHeaderWarnings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const { t } = useTranslation();
  const isEnabled = useEntityIsEnabled(entityIdentifier);
  const selectWarnings = useMemo(() => buildSelectWarnings(entityIdentifier, t), [entityIdentifier, t]);
  const warnings = useAppSelector(selectWarnings);

  if (warnings.length === 0) {
    return null;
  }

  return (
    // Using IconButton here bc it matches the styling of the actual buttons in the header without any fanagling, but
    // it's not a button
    <IconButton
      as="span"
      size="sm"
      variant="link"
      alignSelf="stretch"
      aria-label="warnings"
      tooltip={<TooltipContent warnings={warnings} />}
      icon={<PiWarningBold />}
      colorScheme="warning"
      isDisabled={!isEnabled}
    />
  );
});

CanvasEntityHeaderWarnings.displayName = 'CanvasEntityHeaderWarnings';

const TooltipContent = memo((props: { warnings: string[] }) => {
  const { t } = useTranslation();
  return (
    <Flex flexDir="column">
      <Text>{t('controlLayers.warnings.problemsFound')}:</Text>
      <UnorderedList>
        {props.warnings.map((warning, index) => (
          <ListItem key={index}>{warning}</ListItem>
        ))}
      </UnorderedList>
    </Flex>
  );
});
TooltipContent.displayName = 'TooltipContent';
