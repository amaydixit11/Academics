% ==========================================
% Knowledge Base (Facts)
% ==========================================
is_warm.
is_raining.
is_sunny.
is_pleasant.

% ==========================================
% Derived Rules
% ==========================================
enjoy :- is_sunny, is_warm.
go_strawberry_picking :- is_warm, is_pleasant.
no_strawberry_picking :- is_raining.
wet :- is_raining.

% ==========================================
% Logical Transformation Utilities
% ==========================================
transform_implications(implies(X, Y), or(not(Xt), Yt)) :-
    transform_implications(X, Xt),
    transform_implications(Y, Yt).
transform_implications(and(X, Y), and(Xt, Yt)) :-
    transform_implications(X, Xt),
    transform_implications(Y, Yt).
transform_implications(or(X, Y), or(Xt, Yt)) :-
    transform_implications(X, Xt),
    transform_implications(Y, Yt).
transform_implications(not(X), not(Xt)) :-
    transform_implications(X, Xt).
transform_implications(X, X) :- atomic(X).

% ==========================================
% Demonstration Predicate
% ==========================================
demo_transformations :-
    transform_implications(implies(and(is_sunny, is_warm), enjoy), R1),
    writeln('1) Happy rule:'), writeln(R1),
    transform_implications(implies(and(is_warm, is_pleasant), go_strawberry_picking), R2),
    writeln('2) Strawberry picking rule:'), writeln(R2),
    transform_implications(implies(is_raining, no_strawberry_picking), R3),
    writeln('3) No strawberry picking rule:'), writeln(R3),
    transform_implications(implies(is_raining, wet), R4),
    writeln('4) Wet rule:'), writeln(R4).

% ==========================================
% Clausal Form (CNF)
% ==========================================
cnf_kb([
    [enjoy, not(is_sunny), not(is_warm)],
    [go_strawberry_picking, not(is_warm), not(is_pleasant)],
    [no_strawberry_picking, not(is_raining)],
    [wet, not(is_raining)],
    [is_warm],
    [is_raining],
    [is_sunny],
    [is_pleasant]
]).

% ==========================================
% Utility Predicates for Resolution
% ==========================================
negate_goal(not(X), [X]) :- !.
negate_goal(X, [not(X)]).

opposite(Lit, not(Lit)).
opposite(not(Lit), Lit).

resolve_clauses(C1, C2, Result) :-
    member(Lit, C1),
    opposite(Lit, Opp),
    member(Opp, C2),
    delete(C1, Lit, Temp1),
    delete(C2, Opp, Temp2),
    append(Temp1, Temp2, Temp),
    sort(Temp, Result).

is_empty([]).

% ==========================================
% Resolution Refutation Procedure
% ==========================================
prove(Goal) :-
    cnf_kb(BaseKB),
    negate_goal(Goal, Negated),
    append(BaseKB, [Negated], Clauses),
    writeln('--- Resolution Steps ---'),
    goal_directed_resolution(Clauses, Negated, []).

goal_directed_resolution(Clauses, _, _) :-
    member([], Clauses), !,
    writeln('✅ Refutation successful! Goal PROVED.').

goal_directed_resolution(Clauses, Goal, History) :-
    findall((A, B, R),
        (
            select(A, Clauses, _),
            select(B, Clauses, _),
            A \= B,
            resolve_clauses(A, B, R),
            \+ member(R, Clauses),
            (member(Goal, A); member(Goal, B); member(not(Goal), A); member(not(Goal), B))
        ),
        GoalPairs),
    ( GoalPairs = [] ->
        findall((A, B, R),
            (
                select(A, Clauses, _),
                select(B, Clauses, _),
                A \= B,
                resolve_clauses(A, B, R),
                \+ member(R, Clauses)
            ),
            Pairs),
        ( Pairs = [] ->
            writeln('❌ No contradiction found. Goal NOT provable.'), !
        ; Pairs = [(A, B, R)|_],
          format('Resolved -> ~w~n', [R]),
          append(Clauses, [R], NewClauses),
          goal_directed_resolution(NewClauses, Goal, [(A, B)|History])
        )
    ; GoalPairs = [(A, B, R)|_],
      format('Resolved (goal-related) -> ~w~n', [R]),
      append(Clauses, [R], Updated),
      goal_directed_resolution(Updated, Goal, [(A, B)|History])
    ).
