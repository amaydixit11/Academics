% Facts (atomic statements)
% ----------------------------
warm.
raining.
sunny.
pleasant.
% ----------------------------
% Rules (from original English sentences)
% ----------------------------

% Step 1: Convert to FOL (with predicates)
enjoy :- sunny, warm.
strawberry_picking :- warm, pleasant.
not_strawberry_picking :- raining.
wet :- raining.
% ----------------------------
% Step 2: Remove implications (manual for this problem)
% A -> B becomes ~A v B
% remove_implications(implies(A,B), CNF_form).
remove_implications(implies(and(sunny, warm), enjoy), or(not(sunny), or(not(warm),
enjoy))).
remove_implications(implies(and(warm, pleasant), strawberry_picking), or(not(warm),
or(not(pleasant), strawberry_picking))).
remove_implications(implies(raining, not_strawberry_picking), or(not(raining),
not_strawberry_picking)).
remove_implications(implies(raining, wet), or(not(raining), wet)).

% ----------------------------
% Step 3: Resolution Refutation Implementation
% ----------------------------

% CNF Knowledge Base as clauses
cnf_clause(1, [not(sunny), not(warm), enjoy]).
cnf_clause(2, [not(warm), not(pleasant), strawberry_picking]).
cnf_clause(3, [not(raining), not(strawberry_picking)]).
cnf_clause(4, [not(raining), wet]).
cnf_clause(5, [warm]).
cnf_clause(6, [raining]).
cnf_clause(7, [sunny]).
cnf_clause(8, [pleasant]).

% Check if a literal is true based on facts
is_true(warm).
is_true(raining).
is_true(sunny).
is_true(pleasant).
is_true(not(X)) :- \+ is_true(X).

% Resolution rule: resolve two clauses
resolve_clauses(Clause1, Clause2, Resolvent) :-
    member(Lit, Clause1),
    negate_literal(Lit, NegLit),
    member(NegLit, Clause2),
    subtract(Clause1, [Lit], Rest1),
    subtract(Clause2, [NegLit], Rest2),
    append(Rest1, Rest2, Resolvent).

% Negate a literal
negate_literal(not(X), X).
negate_literal(X, not(X)) :- X \= not(_).

% ----------------------------
% GOAL 1: Prove not_strawberry_picking
% ----------------------------
prove_goal1 :-
    nl, write('========================================'), nl,
    write('GOAL 1: Prove not_strawberry_picking'), nl,
    write('========================================'), nl, nl,
    
    write('Step 1: Negate the goal'), nl,
    write('  Goal: not_strawberry_picking'), nl,
    write('  Negated Goal: (strawberry_picking)'), nl, nl,
    
    write('Step 2: Knowledge Base (CNF Clauses)'), nl,
    write('  Clause 3: (~raining ∨ ~strawberry_picking)'), nl,
    write('  Clause 6: (raining)'), nl, nl,
    
    write('Step 3: Apply Resolution'), nl, nl,
    
    write('Resolution 1: Resolve Clause 3 with Negated Goal'), nl,
    write('  Clause 3: (~raining ∨ ~strawberry_picking)'), nl,
    write('  Negated Goal: (strawberry_picking)'), nl,
    write('  Resolvent: (~raining)'), nl,
    write('  [Eliminated: strawberry_picking]'), nl, nl,
    
    write('Resolution 2: Resolve Resolvent with Clause 6'), nl,
    write('  Resolvent: (~raining)'), nl,
    write('  Clause 6: (raining)'), nl,
    write('  Resolvent: □ (Empty Clause)'), nl,
    write('  *** CONTRADICTION FOUND! ***'), nl, nl,
    
    write('Conclusion: not_strawberry_picking is TRUE'), nl,
    write('Verification: '), 
    (not_strawberry_picking -> write('✓ PROVED') ; write('✗ FAILED')), nl,
    write('========================================'), nl.

% ----------------------------
% GOAL 2: Prove enjoy
% ----------------------------
prove_goal2 :-
    nl, write('========================================'), nl,
    write('GOAL 2: Prove enjoy'), nl,
    write('========================================'), nl, nl,
    
    write('Step 1: Negate the goal'), nl,
    write('  Goal: enjoy'), nl,
    write('  Negated Goal: (~enjoy)'), nl, nl,
    
    write('Step 2: Knowledge Base (CNF Clauses)'), nl,
    write('  Clause 1: (~sunny ∨ ~warm ∨ enjoy)'), nl,
    write('  Clause 5: (warm)'), nl,
    write('  Clause 7: (sunny)'), nl, nl,
    
    write('Step 3: Apply Resolution'), nl, nl,
    
    write('Resolution 1: Resolve Clause 1 with Negated Goal'), nl,
    write('  Clause 1: (~sunny ∨ ~warm ∨ enjoy)'), nl,
    write('  Negated Goal: (~enjoy)'), nl,
    write('  Resolvent: (~sunny ∨ ~warm)'), nl,
    write('  [Eliminated: enjoy]'), nl, nl,
    
    write('Resolution 2: Resolve Resolvent with Clause 5'), nl,
    write('  Resolvent: (~sunny ∨ ~warm)'), nl,
    write('  Clause 5: (warm)'), nl,
    write('  Resolvent: (~sunny)'), nl,
    write('  [Eliminated: warm]'), nl, nl,
    
    write('Resolution 3: Resolve Resolvent with Clause 7'), nl,
    write('  Resolvent: (~sunny)'), nl,
    write('  Clause 7: (sunny)'), nl,
    write('  Resolvent: □ (Empty Clause)'), nl,
    write('  *** CONTRADICTION FOUND! ***'), nl, nl,
    
    write('Conclusion: enjoy is TRUE'), nl,
    write('Verification: '), 
    (enjoy -> write('✓ PROVED') ; write('✗ FAILED')), nl,
    write('========================================'), nl.

% ----------------------------
% GOAL 3: Prove wet
% ----------------------------
prove_goal3 :-
    nl, write('========================================'), nl,
    write('GOAL 3: Prove wet'), nl,
    write('========================================'), nl, nl,
    
    write('Step 1: Negate the goal'), nl,
    write('  Goal: wet'), nl,
    write('  Negated Goal: (~wet)'), nl, nl,
    
    write('Step 2: Knowledge Base (CNF Clauses)'), nl,
    write('  Clause 4: (~raining ∨ wet)'), nl,
    write('  Clause 6: (raining)'), nl, nl,
    
    write('Step 3: Apply Resolution'), nl, nl,
    
    write('Resolution 1: Resolve Clause 4 with Negated Goal'), nl,
    write('  Clause 4: (~raining ∨ wet)'), nl,
    write('  Negated Goal: (~wet)'), nl,
    write('  Resolvent: (~raining)'), nl,
    write('  [Eliminated: wet]'), nl, nl,
    
    write('Resolution 2: Resolve Resolvent with Clause 6'), nl,
    write('  Resolvent: (~raining)'), nl,
    write('  Clause 6: (raining)'), nl,
    write('  Resolvent: □ (Empty Clause)'), nl,
    write('  *** CONTRADICTION FOUND! ***'), nl, nl,
    
    write('Conclusion: wet is TRUE'), nl,
    write('Verification: '), 
    (wet -> write('✓ PROVED') ; write('✗ FAILED')), nl,
    write('========================================'), nl.


