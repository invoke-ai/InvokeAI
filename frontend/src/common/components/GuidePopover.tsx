import {
    Popover,
    PopoverArrow,
    PopoverContent,
    PopoverTrigger,
    Link,
    PopoverHeader,
    Flex,
    Spacer,
    Image,
    Button
  } from "@chakra-ui/react";
import { ReactElement } from "react";
import { ExternalLinkIcon } from "@chakra-ui/icons";
import { Guides } from "../../app/guides";
  /**
   * The GuidePopover needs a child 'children' to be
   * the trigger component you hover on. That child
   * will be of type ReactElement.
   *
   * It also needs a feature name 'feature', which it
   * uses to look up the 'text' and 'href' from 'help.ts'
   */
type GuideProps = {
    children: ReactElement;
    feature: keyof typeof Guides;
  };
  

  const shouldShowGuide = true
  const shouldShowGuideVisuals = true
const Guide = ({ children, feature }: GuideProps) => {
    const { text, href, guideImage } = Guides[feature];
    if (shouldShowGuide) {    
        return (
            <Popover trigger={"hover"}>
                <PopoverTrigger>{children}</PopoverTrigger>
                <PopoverContent width={"auto"}>
                    <PopoverArrow />
                    if (shouldShowGuideVisuals) {   
                        <PopoverHeader>
                            <Flex alignItems={"center"} gap={2}>
                                <Image src={guideImage} alt={text} />
                            </Flex>
                        </PopoverHeader>}
                
                    
                        <Flex alignItems={"center"} gap={2} p={4}>
                            {text} 
                        </Flex>
                        <Flex alignItems={"right"} gap={2} p={4}>
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />   
                            <Link href={href} isExternal>
                                <Button leftIcon={<ExternalLinkIcon />} colorScheme='teal' variant='solid'>
                                Learn More
                                </Button>
                            </Link>
                        </Flex>           
                </PopoverContent>
            </Popover>
            );
        };
return ;
};

    
export default GuidePopover;