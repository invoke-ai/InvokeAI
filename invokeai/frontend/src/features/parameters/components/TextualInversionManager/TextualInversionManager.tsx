import { Box } from '@chakra-ui/react';
import { getTextualInversionTriggers } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISimpleMenu, { IAIMenuItem } from 'common/components/IAISimpleMenu';
import { setTextualInversionsInUse } from 'features/parameters/store/generationSlice';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export default function TextualInversionManager() {
  const dispatch = useAppDispatch();
  const textualInversionsInUse = useAppSelector(
    (state: RootState) => state.generation.textualInversionsInUse
  );
  const foundLocalTextualInversionTriggers = useAppSelector(
    (state) => state.system.foundLocalTextualInversionTriggers
  );
  const foundHuggingFaceTextualInversionTriggers = useAppSelector(
    (state) => state.system.foundHuggingFaceTextualInversionTriggers
  );

  const localTextualInversionTriggers = useAppSelector(
    (state) => state.generation.localTextualInversionTriggers
  );

  const huggingFaceTextualInversionConcepts = useAppSelector(
    (state) => state.generation.huggingFaceTextualInversionConcepts
  );

  const shouldShowHuggingFaceConcepts = useAppSelector(
    (state) => state.ui.shouldShowHuggingFaceConcepts
  );

  const { t } = useTranslation();

  useEffect(() => {
    dispatch(getTextualInversionTriggers());
  }, [dispatch]);

  const handleTextualInversion = (textual_inversion: string) => {
    dispatch(setTextualInversionsInUse(textual_inversion));
  };

  const renderTextualInversionOption = (textual_inversion: string) => {
    const thisTIExists = textualInversionsInUse.includes(textual_inversion);
    const tiExistsStyle = {
      fontWeight: 'bold',
      color: 'var(--context-menu-active-item)',
    };
    return (
      <Box style={thisTIExists ? tiExistsStyle : {}}>{textual_inversion}</Box>
    );
  };

  const numOfActiveTextualInversions = () => {
    const allTextualInversions = localTextualInversionTriggers.concat(
      huggingFaceTextualInversionConcepts
    );
    return allTextualInversions.filter((ti) =>
      textualInversionsInUse.includes(ti)
    ).length;
  };

  const makeTextualInversionItems = () => {
    const textualInversionsFound: IAIMenuItem[] = [];
    foundLocalTextualInversionTriggers?.forEach((textualInversion) => {
      if (textualInversion.name !== ' ') {
        const newTextualInversionItem: IAIMenuItem = {
          item: renderTextualInversionOption(textualInversion.name),
          onClick: () => handleTextualInversion(textualInversion.name),
        };
        textualInversionsFound.push(newTextualInversionItem);
      }
    });

    if (shouldShowHuggingFaceConcepts) {
      foundHuggingFaceTextualInversionTriggers?.forEach((textualInversion) => {
        if (textualInversion.name !== ' ') {
          const newTextualInversionItem: IAIMenuItem = {
            item: renderTextualInversionOption(textualInversion.name),
            onClick: () => handleTextualInversion(textualInversion.name),
          };
          textualInversionsFound.push(newTextualInversionItem);
        }
      });
    }

    return textualInversionsFound;
  };

  return foundLocalTextualInversionTriggers &&
    (foundLocalTextualInversionTriggers?.length > 0 ||
      (foundHuggingFaceTextualInversionTriggers &&
        foundHuggingFaceTextualInversionTriggers?.length > 0 &&
        shouldShowHuggingFaceConcepts)) ? (
    <IAISimpleMenu
      menuItems={makeTextualInversionItems()}
      menuType="regular"
      buttonText={`${t(
        'modelManager.addTextualInversionTrigger'
      )} (${numOfActiveTextualInversions()})`}
    />
  ) : (
    <Box
      background="var(--btn-base-color)"
      padding={2}
      textAlign="center"
      borderRadius={4}
      fontWeight="bold"
    >
      {t('modelManager.noTextualInversionTriggers')}
    </Box>
  );
}
