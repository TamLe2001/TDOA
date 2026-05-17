Rafael Moura Belling Christiansen \
AU750489

Jeg har primært stået for det matematiske fundament i projektet. Min opgave var at gennemskue TDOA-logikken fra den oprindelige Python-kode og få den oversat til C++. Det indebar især at implementere trilateration-formlerne og det "brute-force" loop, som systematisk tester forskellige scale-faktorer for at finde frem til lydkildens præcise position.

Projektet viste mig at nogle python, kan hurtigt blive tungt for computere at trække, dette skyldes i dette tilfæjde primært 2 ting. Det første problem, Biblioteket der blev brugt Cliford.CGA, udnytter multivektore med mange dimentioner til deres udregninger, som gør at hver udregninger tager en del computer kraft at gennemføre. Det andet problem er at python er et fortolket sprog, som derfor kraver hvis programmet man laver skal gøre hurtigt, skal der bruges ektra tid på optimesering.

Hvis jeg skulle have ændret noget, ville jeg nok have kigget på en mere elegant søgemetode end det lineære loop, måske en binary search for at optimere hastigheden endnu mere.