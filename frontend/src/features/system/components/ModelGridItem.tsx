import { 
  Card,
  CardBody,
  Image,
  Heading,
  Stack,
  Text
} from '@chakra-ui/react'
import { RootState } from 'app/store';
import React from 'react';
import { useAppSelector } from 'app/storeHooks';

export default function ModelGridItem({model, isDisabled, isSelected, onSelect}) {
  const { isProcessing, isConnected } = useAppSelector(
    (state: RootState) => state.system
  );

  const openModel = useAppSelector(
    (state: RootState) => state.system.openModel
  );

  const { name } = model;

  return (
    <Card
      borderColor='var(--background-color)'
      borderWidth='1px'
      height='300px'
      maxH="300px"
      backgroundColor={isSelected ? 'var(--accent-color)' : ''}
      minW='33%'
      _hover={{
        backgroundColor:
          name === openModel
            ? 'var(--accent-color)'
            : 'var(--background-color)',
      }}
      overflow='hidden'
      cursor='pointer'
      onClick={() => onSelect(name)}
    >
      {model.image && <Image
          objectFit='cover'
          align='top'
          maxH='120px'
          src={model.image}
        />}
      <CardBody>          
        <Stack mt='1' spacing='1'>
          <Heading size="md">{model.name}</Heading>
          {/* {model.url && <Text fontSize="xs"><a href={model.url} target="_blank">{model.url}</a></Text>} */}
          {model.url && <Text fontSize="xs">{model.url}</Text>}
          {model.tags && <Text>{model.tags}</Text>}
          {model.markdown && <Text><strong>{model.markdown}</strong></Text>}
          <Text fontSize='xs'>{model.description}</Text>
        </Stack>
      </CardBody>
    </Card>
  );
}
