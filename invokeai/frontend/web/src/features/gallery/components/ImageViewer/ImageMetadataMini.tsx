import { Alert, AlertIcon, AlertTitle, Grid, GridItem, Text } from '@invoke-ai/ui-library';
import { useMetadataItem } from 'features/metadata/hooks/useMetadataItem';
import { handlers } from 'features/metadata/util/handlers';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

export const ImageMetadataMini = ({ imageName }: { imageName: string }) => {
  const { metadata, isLoading } = useDebouncedMetadata(imageName);
  const createdBy = useMetadataItem(metadata, handlers.createdBy);
  const positivePrompt = useMetadataItem(metadata, handlers.positivePrompt);
  const negativePrompt = useMetadataItem(metadata, handlers.negativePrompt);
  const seed = useMetadataItem(metadata, handlers.seed);
  const model = useMetadataItem(metadata, handlers.model);
  const strength = useMetadataItem(metadata, handlers.strength);

  if (isLoading) {
    return (
      <Alert status="loading" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
        <AlertIcon />
        <AlertTitle>Loading metadata...</AlertTitle>
      </Alert>
    );
  }
  if (
    !createdBy.valueOrNull &&
    !positivePrompt.valueOrNull &&
    !negativePrompt.valueOrNull &&
    !seed.valueOrNull &&
    !model.valueOrNull &&
    !strength.valueOrNull
  ) {
    return (
      <Alert status="warning" borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
        <AlertTitle>No metadata found</AlertTitle>
      </Alert>
    );
  }
  return (
    <Alert borderRadius="base" fontSize="sm" shadow="md" w="fit-content">
      <Grid gridTemplateColumns="auto 1fr" columnGap={2} maxW={420}>
        {createdBy.valueOrNull && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{createdBy.label}:</Text>
            </GridItem>
            <GridItem>{createdBy.renderedValue}</GridItem>
          </>
        )}
        {positivePrompt.valueOrNull && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{positivePrompt.label}:</Text>
            </GridItem>
            <GridItem>{positivePrompt.renderedValue}</GridItem>
          </>
        )}
        {negativePrompt.valueOrNull && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{negativePrompt.label}:</Text>
            </GridItem>
            <GridItem>{negativePrompt.renderedValue}</GridItem>
          </>
        )}
        {model.valueOrNull !== null && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{model.label}:</Text>
            </GridItem>
            <GridItem>{model.renderedValue}</GridItem>
          </>
        )}
        {strength.valueOrNull !== null && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{strength.label}:</Text>
            </GridItem>
            <GridItem>{strength.renderedValue}</GridItem>
          </>
        )}
        {seed.valueOrNull !== null && (
          <>
            <GridItem textAlign="end">
              <Text fontWeight="semibold">{seed.label}:</Text>
            </GridItem>
            <GridItem>{seed.renderedValue}</GridItem>
          </>
        )}
      </Grid>
    </Alert>
  );
};
ImageMetadataMini.displayName = 'ImageMetadataMini';
