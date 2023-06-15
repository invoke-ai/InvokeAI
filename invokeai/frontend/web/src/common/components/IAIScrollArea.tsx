import { ScrollArea, ScrollAreaProps } from '@mantine/core';

type IAIScrollArea = ScrollAreaProps;

export default function IAIScrollArea(props: IAIScrollArea) {
  const { ...rest } = props;
  return (
    <ScrollArea w="100%" {...rest}>
      {props.children}
    </ScrollArea>
  );
}
