import { Box, Grid, Text } from '@chakra-ui/react';

interface HotkeysModalProps {
  hotkey: string;
  title: string;
  description?: string;
}

export default function HotkeysModalItem(props: HotkeysModalProps) {
  const { title, hotkey, description } = props;
  return (
    <Grid
      sx={{
        gridTemplateColumns: 'auto max-content',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}
    >
      <Grid>
        <Text fontWeight={600}>{title}</Text>
        {description && (
          <Text
            sx={{
              fontSize: 'sm',
            }}
            variant="subtext"
          >
            {description}
          </Text>
        )}
      </Grid>
      <Box
        sx={{
          fontSize: 'sm',
          fontWeight: 600,
          px: 2,
          py: 1,
        }}
      >
        {hotkey}
      </Box>
    </Grid>
  );
}
