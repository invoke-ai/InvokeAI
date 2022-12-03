import { Box, Flex, VStack } from '@chakra-ui/react';
import { searchForModels } from 'app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import React, { ReactNode, ChangeEvent } from 'react';
import { MdFindInPage } from 'react-icons/md';
import _ from 'lodash';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaPlus } from 'react-icons/fa';
import {
  setFoundModels,
  setSearchFolder,
} from 'features/system/store/systemSlice';

export default function SearchModels() {
  const dispatch = useAppDispatch();

  const searchFolder = useAppSelector(
    (state: RootState) => state.system.searchFolder
  );

  const foundModels = useAppSelector(
    (state: RootState) => state.system.foundModels
  );

  const [modelsToAdd, setModelsToAdd] = React.useState<string[]>([]);

  const resetSearchModelHandler = () => {
    dispatch(setSearchFolder(null));
    dispatch(setFoundModels(null));
    setModelsToAdd([]);
  };

  const findModelsHandler = () => {
    dispatch(searchForModels());
  };

  const foundModelsChangeHandler = (e: ChangeEvent<HTMLInputElement>) => {
    if (!modelsToAdd.includes(e.target.value)) {
      setModelsToAdd([...modelsToAdd, e.target.value]);
    } else {
      setModelsToAdd(_.remove(modelsToAdd, (v) => v !== e.target.value));
    }
  };

  const addAllToSelected = () => {
    setModelsToAdd([]);
    if (foundModels) {
      foundModels.forEach((model) => {
        setModelsToAdd((currentModels) => {
          return [...currentModels, model.name];
        });
      });
    }
  };

  const removeAllFromSelected = () => {
    setModelsToAdd([]);
  };

  const renderFoundModels = () => {
    const foundModelsToRender: ReactNode[] = [];

    if (foundModels) {
      foundModels.forEach((model, index) => {
        foundModelsToRender.push(
          <Box key={index}>
            <IAICheckbox
              value={model.name}
              label={
                <>
                  <VStack alignItems={'start'}>
                    <p style={{ fontWeight: 'bold' }}>{model.name}</p>
                    <p style={{ fontStyle: 'italic' }}>{model.location}</p>
                  </VStack>
                </>
              }
              isChecked={modelsToAdd.includes(model.name)}
              onChange={foundModelsChangeHandler}
              padding={'1rem'}
              backgroundColor={'var(--background-color)'}
              borderRadius={'0.5rem'}
              _checked={{
                backgroundColor: 'var(--accent-color)',
                color: 'var(--text-color)',
              }}
            ></IAICheckbox>
          </Box>
        );
      });
    }

    return foundModelsToRender;
  };

  return (
    <>
      {searchFolder ? (
        <Flex
          flexDirection={'column'}
          padding={'1rem'}
          backgroundColor={'var(--background-color)'}
          borderRadius="0.5rem"
          rowGap={'0.5rem'}
          position={'relative'}
        >
          <p
            style={{
              fontWeight: 'bold',
              fontSize: '0.8rem',
              backgroundColor: 'var(--background-color-secondary)',
              padding: '0.2rem 1rem',
              width: 'max-content',
              borderRadius: '0.2rem',
            }}
          >
            Checkpoint Folder
          </p>
          <p style={{ fontWeight: 'bold' }}>{searchFolder}</p>
          <IAIIconButton
            aria-label="Clear Checkpoint Folder"
            icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
            position={'absolute'}
            right={5}
            onClick={resetSearchModelHandler}
          />
        </Flex>
      ) : (
        <IAIButton aria-label="Find Models" onClick={findModelsHandler}>
          <Flex columnGap={'0.5rem'}>
            <MdFindInPage fontSize={20} />
            Select Folder
          </Flex>
        </IAIButton>
      )}
      {foundModels && (
        <Flex flexDirection={'column'} rowGap={'1rem'}>
          <Flex justifyContent={'space-between'} alignItems="center">
            <p>Models Found: {foundModels.length}</p>
            <p>Selected: {modelsToAdd.length}</p>
          </Flex>
          <Flex columnGap={'0.5rem'} justifyContent={'space-between'}>
            <Flex columnGap={'0.5rem'}>
              <IAIButton
                isDisabled={modelsToAdd.length === foundModels.length}
                onClick={addAllToSelected}
              >
                Select All
              </IAIButton>
              <IAIButton
                isDisabled={modelsToAdd.length === 0}
                onClick={removeAllFromSelected}
              >
                Deselect All
              </IAIButton>
            </Flex>

            <IAIButton isDisabled={modelsToAdd.length === 0}>
              Add Selected
            </IAIButton>
          </Flex>
          <Flex
            rowGap={'1rem'}
            flexDirection="column"
            maxHeight={'20rem'}
            overflowY="scroll"
            paddingRight={'1rem'}
            paddingLeft={'0.2rem'}
          >
            {renderFoundModels()}
          </Flex>
        </Flex>
      )}
    </>
  );
}
