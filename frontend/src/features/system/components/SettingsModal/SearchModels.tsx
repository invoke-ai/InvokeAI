import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  Input,
  VStack,
} from '@chakra-ui/react';
import { searchForModels } from 'app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import { Field, Formik } from 'formik';
import React from 'react';
import { MdFindInPage } from 'react-icons/md';
import _ from 'lodash';
import IAIButton from 'common/components/IAIButton';

export default function SearchModels() {
  const dispatch = useAppDispatch();
  const foundModels = useAppSelector(
    (state: RootState) => state.system.foundModels
  );

  const [modelsToAdd, setModelsToAdd] = React.useState([]);

  const findModelsHandler = (values: any) => {
    dispatch(searchForModels(values.search_folder));
  };

  const foundModelsChangeHandler = (e) => {
    if (!modelsToAdd.includes(e.target.value)) {
      setModelsToAdd([...modelsToAdd, e.target.value]);
    } else {
      setModelsToAdd(_.remove(modelsToAdd, (v) => v !== e.target.value));
    }
  };

  const addAllToSelected = () => {
    setModelsToAdd([]);
    foundModels.forEach((model) => {
      setModelsToAdd((currentModels) => {
        return [...currentModels, model.name];
      });
    });
  };

  const removeAllFromSelected = () => {
    setModelsToAdd([]);
  };

  const renderFoundModels = () => {
    const foundModelsToRender = [];

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
      <Formik
        initialValues={{ search_folder: '' }}
        onSubmit={findModelsHandler}
      >
        {({ handleSubmit }) => (
          <form onSubmit={handleSubmit}>
            <HStack alignItems={'center'} columnGap="0.5rem">
              {/* Search Folder */}
              <FormControl isRequired>
                <FormLabel htmlFor="search_folder">Search Folder</FormLabel>
                <Field
                  as={Input}
                  id="search_folder"
                  name="search_folder"
                  type="text"
                />
              </FormControl>
              <Box paddingTop={'2rem'}>
                <IAIIconButton
                  aria-label="Find Models"
                  tooltip="Find Models"
                  icon={<MdFindInPage />}
                  type="submit"
                />
              </Box>
            </HStack>
          </form>
        )}
      </Formik>
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
            maxHeight={'24rem'}
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
