import { Box, Flex } from '@chakra-ui/react';
import { getTextualInversionTriggers } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISimpleMenu, { IAIMenuItem } from 'common/components/IAISimpleMenu';
import {
  setAddTIToNegative,
  setClearTextualInversions,
  setTextualInversionsInUse,
} from 'features/parameters/store/generationSlice';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { MdArrowDownward, MdClear } from 'react-icons/md';

export default function TextualInversionManager() {
  const dispatch = useAppDispatch();
  const textualInversionsInUse = useAppSelector(
    (state: RootState) => state.generation.textualInversionsInUse
  );

  const negativeTextualInversionsInUse = useAppSelector(
    (state: RootState) => state.generation.negativeTextualInversionsInUse
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

  const addTIToNegative = useAppSelector(
    (state) => state.generation.addTIToNegative
  );

  const { t } = useTranslation();

  useEffect(() => {
    dispatch(getTextualInversionTriggers());
  }, [dispatch]);

  const handleTextualInversion = (textual_inversion: string) => {
    dispatch(setTextualInversionsInUse(textual_inversion));
  };

  const TIPip = ({ color }: { color: string }) => {
    return (
      <Box width={2} height={2} borderRadius={9999} backgroundColor={color}>
        {' '}
      </Box>
    );
  };

  const renderTextualInversionOption = (textual_inversion: string) => {
    return (
      <Flex alignItems="center" columnGap={1}>
        {textual_inversion}
        {textualInversionsInUse.includes(textual_inversion) && (
          <TIPip color="var(--context-menu-active-item)" />
        )}
        {negativeTextualInversionsInUse.includes(textual_inversion) && (
          <TIPip color="var(--status-bad-color)" />
        )}
      </Flex>
    );
  };

  const numOfActiveTextualInversions = () => {
    const allTextualInversions = localTextualInversionTriggers.concat(
      huggingFaceTextualInversionConcepts
    );
    return allTextualInversions.filter(
      (ti) =>
        textualInversionsInUse.includes(ti) ||
        negativeTextualInversionsInUse.includes(ti)
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
    <Flex columnGap={2}>
      <IAISimpleMenu
        menuItems={makeTextualInversionItems()}
        menuType="regular"
        buttonText={`${t(
          'modelManager.addTextualInversionTrigger'
        )} (${numOfActiveTextualInversions()})`}
        menuButtonProps={{
          width: '100%',
          padding: '0 1rem',
        }}
      />
      <IAIIconButton
        icon={<MdArrowDownward />}
        style={{
          backgroundColor: addTIToNegative ? 'var(--btn-delete-image)' : '',
        }}
        tooltip={t('modelManager.addTIToNegative')}
        aria-label={t('modelManager.addTIToNegative')}
        onClick={() => dispatch(setAddTIToNegative(!addTIToNegative))}
      />
      <IAIIconButton
        icon={<MdClear />}
        tooltip={t('modelManager.clearTextualInversions')}
        aria-label={t('modelManager.clearTextualInversions')}
        onClick={() => dispatch(setClearTextualInversions())}
      />
    </Flex>
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
