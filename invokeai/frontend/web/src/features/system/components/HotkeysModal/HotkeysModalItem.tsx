import { Box, Grid } from '@chakra-ui/react';
import { InvText } from 'common/components/InvText/wrapper';

interface HotkeysModalProps {
  hotkey: string;
  title: string;
  description?: string;
}

export default function HotkeysModalItem(props: HotkeysModalProps) {
  const { title, hotkey, description } = props;
  return (
    <Grid
      gridTemplateColumns="auto max-content"
      justifyContent="space-between"
      alignItems="center"
    >
      <Grid>
        <InvText fontWeight="semibold">{title}</InvText>
        {description && <InvText variant="subtext">{description}</InvText>}
      </Grid>
      <Box fontSize="sm" fontWeight="semibold" px={2} py={1}>
        {hotkey}
      </Box>
    </Grid>
  );
}
