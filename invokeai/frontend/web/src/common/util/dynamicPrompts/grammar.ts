import * as ohm from 'ohm-js';

const grammarSource = `
Prompt {
  exp
    = exp choicesStart multi_number multi_trigger choices choicesEnd exp --multi_choices
    | exp choicesStart multi_number multi_trigger multi_joinString multi_trigger choices choicesEnd exp --multi_choicesWithJoinString
    | exp choicesStart choices choicesEnd exp --choices
    | text --text

  choices
    = listOf<choice, choicesDelimiter>

  text
  	= (~choicesStart any)* 

  choice
    = (~reservedChar any)*

  reservedChar
    = (choicesEnd | choicesStart | choicesDelimiter | multi_trigger)

  choicesStart
    = "{"

  choicesEnd
    = "}"

  choicesDelimiter
    = "|"
    
  multi_trigger
    = "$$"
    
  multi_number
    = digit+
    
  multi_joinString
    = (~multi_trigger choice)
  
}`;

const getPermutationsOfSize = (array: string[], size: number): string[][] => {
  const result: string[][] = [];

  function permute(arr: string[], m: string[] = []) {
    if (m.length === size) {
      result.push(m);
      return;
    }
    for (let i = 0; i < arr.length; i++) {
      const curr = arr.slice();
      const next = curr.splice(i, 1);
      permute(curr, m.concat(next));
    }
  }

  permute(array);

  return result;
};

export const dynamicPromptsGrammar = ohm.grammar(grammarSource);

export const dynamicPromptsSemantics = dynamicPromptsGrammar
  .createSemantics()
  .addOperation('expand', {
    exp_multi_choices: function (
      before,
      _choicesStart,
      number,
      _multiTrigger,
      choices,
      _choicesEnd,
      after
    ) {
      const beforePermutations = before.expand();
      const choicePermutations = getPermutationsOfSize(
        choices.expand(),
        Number(number.sourceString)
      );
      const afterPermutations = after.expand();
      const sep = ',';

      const combined = [];
      for (const b of beforePermutations) {
        for (const c of choicePermutations) {
          for (const a of afterPermutations) {
            combined.push(b + c.join(sep) + a);
          }
        }
      }
      return combined;
    },
    exp_multi_choicesWithJoinString: function (
      before,
      _choicesStart,
      number,
      _multiTrigger1,
      _multiJoinString,
      _multiTrigger2,
      choices,
      _choicesEnd,
      after
    ) {
      const beforePermutations = before.expand();
      const choicePermutations = getPermutationsOfSize(
        choices.expand(),
        Number(number.sourceString)
      );
      const afterPermutations = after.expand();
      const sep = _multiJoinString.sourceString;

      const combined = [];
      for (const b of beforePermutations) {
        for (const c of choicePermutations) {
          for (const a of afterPermutations) {
            combined.push(b + c.join(sep) + a);
          }
        }
      }
      return combined;
    },
    exp_choices: function (before, _choicesStart, choices, _choicesEnd, after) {
      const beforePermutations = before.expand();
      const choicePermutations = choices.expand();
      const afterPermutations = after.expand();

      const combined: string[] = [];
      for (const b of beforePermutations) {
        for (const c of choicePermutations) {
          for (const a of afterPermutations) {
            combined.push(b + c + a);
          }
        }
      }

      return combined;
    },
    exp_text: function (text) {
      return [text.sourceString];
    },
    choices: function (choices) {
      return choices.asIteration().children.map((c) => c.sourceString);
    },
    _iter: function (...children) {
      children.map((c) => c.expand());
    },
    _terminal: function () {
      return this.sourceString;
    },
  });
