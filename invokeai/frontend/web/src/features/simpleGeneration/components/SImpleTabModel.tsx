import { Flex, FormLabel, Icon, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { modelChanged, selectModel } from 'features/simpleGeneration/store/slice';
import { isModel } from 'features/simpleGeneration/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { MdMoneyOff } from 'react-icons/md';

export const SimpleTabModel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectModel);
  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isModel(e.target.value)) {
        return;
      }
      dispatch(modelChanged({ model: e.target.value }));
    },
    [dispatch]
  );

  return (
    <Flex alignItems="center" gap={2}>
      <InformationalPopover feature="paramModel">
        <FormLabel m={0}>{t('modelManager.model')}</FormLabel>
      </InformationalPopover>
      {model === 'flux' && (
        <InformationalPopover feature="fluxDevLicense" hideDisable={true}>
          <Flex justifyContent="flex-start">
            <Icon as={MdMoneyOff} />
          </Flex>
        </InformationalPopover>
      )}
      <Select value={model} onChange={onChange}>
        <option value="chatgpt-4o">ChatGPT 4o</option>
        <option value="flux">FLUX</option>
        <option value="sdxl">SDXL</option>
        <option value="sd-1">SD 1.5</option>
      </Select>
    </Flex>
  );
});
SimpleTabModel.displayName = 'SimpleTabModel';

// export const SimpleTabModel = memo(() => {
//   const { t } = useTranslation();
//   const dispatch = useAppDispatch();
//   const [modelConfigs] = useSimpleTabModels();
//   const selectedModelConfig = useSimpleTabModelConfig();
//   const onChange = useCallback(
//     (modelConfig: AnyModelConfig) => {
//       dispatch(modelChanged({ model: zModelIdentifierField.parse(modelConfig) }));
//     },
//     [dispatch]
//   );

//   const isFluxDevSelected = useMemo(
//     () =>
//       selectedModelConfig &&
//       isCheckpointMainModelConfig(selectedModelConfig) &&
//       selectedModelConfig.config_path === 'flux-dev',
//     [selectedModelConfig]
//   );

//   return (
//     <Flex alignItems="center" gap={2}>
//       <InformationalPopover feature="paramModel">
//         <FormLabel>{t('modelManager.model')}</FormLabel>
//       </InformationalPopover>
//       {isFluxDevSelected && (
//         <InformationalPopover feature="fluxDevLicense" hideDisable={true}>
//           <Flex justifyContent="flex-start">
//             <Icon as={MdMoneyOff} />
//           </Flex>
//         </InformationalPopover>
//       )}
//       <ModelPicker modelConfigs={modelConfigs} selectedModelConfig={selectedModelConfig} onChange={onChange} grouped />
//     </Flex>
//   );
// });
// SimpleTabModel.displayName = 'SimpleTabModel';
