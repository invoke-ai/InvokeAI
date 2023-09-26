These tests are placed in an "x_" folder so that they are run after
the node tests. If they run beforehand the nodes tests blow up. I
suspect that there are conflicts arising from the in-memory
InvokeAIAppConfig object, but even when I take care to create a fresh
object each time, the problem persists, so perhaps it is something
else?

- Lincoln
