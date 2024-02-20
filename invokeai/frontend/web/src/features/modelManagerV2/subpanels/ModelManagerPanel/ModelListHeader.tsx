import { Box, Divider, Text } from '@invoke-ai/ui-library';

export const ModelListHeader = ({ title }: { title: string }) => {
  return (
    <Box position="relative" padding="10px 0">
      <Divider sx={{ backgroundColor: 'base.400' }} />
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: 0,
          transform: 'translate(0, -50%)',
          backgroundColor: 'base.800',
          padding: '10px',
        }}
      >
        <Text variant="subtext" fontSize="sm">
          {title}
        </Text>
      </Box>
    </Box>
  );
};
