import { ScrollArea, ScrollAreaProps } from '@mantine/core';

type IAIScrollArea = ScrollAreaProps;

export default function IAIScrollArea(props: IAIScrollArea) {
  const { offsetScrollbars = true, ...rest } = props;
  return (
    <ScrollArea w="100%" offsetScrollbars={offsetScrollbars} {...rest}>
      {props.children}
    </ScrollArea>
  );
}
