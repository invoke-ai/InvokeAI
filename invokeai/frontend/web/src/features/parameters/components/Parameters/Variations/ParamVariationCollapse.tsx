// TODO: variations

// import { Flex } from '@chakra-ui/react';
// import { createSelector } from '@reduxjs/toolkit';
// import { stateSelector } from 'app/store/store';
// import { useAppSelector } from 'app/store/storeHooks';
// import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
// import IAICollapse from 'common/components/IAICollapse';
// import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
// import { memo } from 'react';
// import { useTranslation } from 'react-i18next';
// import ParamVariationAmount from './ParamVariationAmount';
// import { ParamVariationToggle } from './ParamVariationToggle';
// import ParamVariationWeights from './ParamVariationWeights';

// const selector = createSelector(
//   stateSelector,
//   (state) => {
//     const activeLabel = state.generation.shouldGenerateVariations
//       ? 'Enabled'
//       : undefined;

//     return { activeLabel };
//   },
//   defaultSelectorOptions
// );

// const ParamVariationCollapse = () => {
//   const { t } = useTranslation();
//   const { activeLabel } = useAppSelector(selector);

//   const isVariationEnabled = useFeatureStatus('variation').isFeatureEnabled;

//   if (!isVariationEnabled) {
//     return null;
//   }

//   return (
//     <IAICollapse label={t('parameters.variations')} activeLabel={activeLabel}>
//       <Flex sx={{ gap: 2, flexDirection: 'column' }}>
//         <ParamVariationToggle />
//         <ParamVariationAmount />
//         <ParamVariationWeights />
//       </Flex>
//     </IAICollapse>
//   );
// };

// export default memo(ParamVariationCollapse);

export default {};
