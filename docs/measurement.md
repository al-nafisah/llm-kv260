# Attested and Cited numbers never blend

Every number this repository reports is one of two kinds.

- **Attested**: measured here, on this board, with the date and the raw log
  committed beside it in [buildlog/](buildlog).
- **Cited**: taken from a paper, a README or an issue thread, and always
  attributed.

The two go in separate tables and are never combined arithmetically or ranked
against each other. Blended, they leave a reader unable to tell which claims
were ever checked here.

## Consequences

A failed run is a result and is logged as one.

A number whose provenance cannot be established is not reported, not even
hedged. A power figure read off a datasheet is a Cited number about the
datasheet, not a measurement of the board.
