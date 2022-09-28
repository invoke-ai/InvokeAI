import { IconButtonProps, IconButton, Tooltip } from '@chakra-ui/react';

interface Props extends IconButtonProps {
  tooltip?: string;
}

/**
 * Reusable customized button component. Originally was more customized - now probably unecessary.
 *
 * TODO: Get rid of this.
 */
const SDIconButton = (props: Props) => {
  const { tooltip = '', ...rest } = props;
  return (
    <Tooltip label={tooltip}>
      <IconButton {...rest} />
    </Tooltip>
  );
};

export default SDIconButton;
