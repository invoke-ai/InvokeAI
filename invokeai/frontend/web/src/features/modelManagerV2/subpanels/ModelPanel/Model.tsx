import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { selectSelectedModelKey, selectSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiExclamationMarkBold } from 'react-icons/pi';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';

import { ModelEdit } from './ModelEdit';
import { ModelView } from './ModelView';

export const Model = memo(() => {
  const { t } = useTranslation();
  const selectedModelMode = useAppSelector(selectSelectedModelMode);
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const { data: modelConfigs, isLoading } = useGetModelConfigsQuery();
  const modelConfig = useMemo(() => {
    if (!modelConfigs) {
      return null;
    }
    if (selectedModelKey === null) {
      return null;
    }
    const modelConfig = modelConfigsAdapterSelectors.selectById(modelConfigs, selectedModelKey);

    if (!modelConfig) {
      return null;
    }

    return modelConfig;
  }, [modelConfigs, selectedModelKey]);

  if (isLoading) {
    return <IAINoContentFallbackWithSpinner label={t('common.loading')} />;
  }

  if (!modelConfig) {
    return <IAINoContentFallback label={t('common.somethingWentWrong')} icon={PiExclamationMarkBold} />;
  }

  if (selectedModelMode === 'view') {
    return <ModelView modelConfig={modelConfig} />;
  }

  return <ModelEdit modelConfig={modelConfig} />;
});

Model.displayName = 'Model';
