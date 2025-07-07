% ======================================================
%              SECTION 1: BASE DEFINITIONS
% ======================================================

% Symmetric edge definitions
e(S1,S2) :- edge(S1,S2,_).
e(S2,S1) :- e(S1,S2).

% Node coloring definitions
node(S1, gray) :- node(S1, C1), edge(S1,S2,C1), edge(S1,S3,C2), C1 != C2.
node(S1, C) :- node(S1, gray), edge(S1, _, C).

% Line-node relationships
hasNodes(C,N) :- line(C), node(N,C).

% ======================================================
%              SECTION 2: PATH SELECTION LOGIC
% ======================================================

% Shortest path (sp) definition
sp((N,N')) :- shortestPath(T), station(T-1,N), station(T-1,N'), N < N'.

% Path selection constraints
0 { selected((N,N'),S1,S2); selected((N,N'),S2,S1) } 1 :- edge(S1,S2,_), sp((N,N')).

% Path incremental definition (improved transitive closure)
path((N,N'), X, Y) :- selected((N,N'), X, Y).
path((N,N'), X, Z) :- path((N,N'), X, Y), selected((N,N'), Y, Z).

% Ensure path connects start to finish
:- sp((N,N')), not path((N,N'),N,N').

% ======================================================
%              SECTION 3: COST CALCULATION
% ======================================================

cost((N,N'), C) :- sp((N,N')), C = #count{X,Y : selected((N,N'),X,Y)}.

#minimize { C,N,N' : cost((N,N'), C) }.

% ======================================================
%              SECTION 4: HOP COMPUTATIONS
% ======================================================

% Generalized hops definition
hops(N,0) :- withinHops(T,_), station(T-1,N).
hops(N,H+1) :- hops(S,H), e(S,N), not hops(N,H), H < 10.

% Minimum hops calculation
minHops(N,V) :- node(N,_), withinHops(_,_), V = #min{X : hops(N,X)}.

% Counting nodes between stations
countedNodes(T+1, C-1) :- countNodesBetween(T), sp((N,N')), cost((N,N'), C).

% ======================================================
%              SECTION 5: SPECIAL PATH QUERIES
% ======================================================

% Paths query
p((N,N')) :- paths(T), station(T-1,N), station(T-1,N'), N < N'.

% Cycle query
sp((N,N)) :- cycle(T), station(T-1,N).
cycleAns(T+1,true) :- cycle(T), path((N,N),N,N).

% Adjacent stations queries
adjacentAns(T+1,true)  :- adjacent(T), station(T-1,N), station(T-1,N'), N < N', e(N,N').
adjacentAns(T+1,false) :- adjacent(T), not adjacentAns(T+1,true).
adjacentToAns(T+1,X)   :- adjacentTo(T), station(T-1,N), station(T-1,N'), N < N', e(N,X), e(N',X).

% Same line query
sameLineAns(T+1,true)  :- sameLine(T), station(T-1,N), station(T-1,N'), N<N', hasNodes(L,N), hasNodes(L,N').
sameLineAns(T+1,false) :- sameLine(T), not sameLineAns(T+1,true).

% Common station queries
sp((N,N')) :- commonStation(T), station(T-1,N), station(T-1,N'), N<N'.
commonStationAns(T+1,true)  :- commonStation(T), cost((N,N'),2).
commonStationAns(T+1,false) :- commonStation(T), not commonStationAns(T+1,true).

% Line-on queries
lineOnAns(T+1,LN) :- linesOnNames(T), station(T-1,N), hasNodes(LN,N).
lineOnAns(T+1,C)  :- linesOnCount(T), station(T-1,N), C = #count{LNAME : hasNodes(LNAME,N), LNAME!=gray}.

% ======================================================
%              SECTION 6: EXISTENCE & STATION QUERIES
% ======================================================

existAns(T+1,true)  :- exist(T), station(T-1,N), node(N,_).
existAns(T+1,false) :- exist(T), not existAns(T+1,true).

stationsAns(T+1,N)  :- stations(T), line(T-1,LN), hasNodes(LN,N).

% ======================================================
%              SECTION 7: PATH & REACHABILITY CONSTRAINTS
% ======================================================

start(S1)  :- p((S1,_)).
finish(S2) :- p((_,S2)).

% Flexible path inclusion constraints (no linearity enforced)
{ in(X,Y) : e(X,Y) }.

% Start and finish constraints
:- start(X), not in(X,_).
:- finish(X), not in(_,X).

% Reachability constraints
reach(X) :- start(X), in(X,_).
reach(Y) :- reach(X), in(X,Y).

:- finish(X), not reach(X).
:- in(X,Y), not reach(X).

% ======================================================
%              SECTION 8: ANSWER COLLECTION
% ======================================================

ans(V) :- end(T), countedNodes(T,V).
ans(V) :- end(T), cycleAns(T,V).
ans(V) :- end(T), adjacentAns(T,V).
ans(V) :- end(T), adjacentToAns(T,V).
ans(V) :- end(T), sameLineAns(T,V).
ans(V) :- end(T), commonStationAns(T,V).
ans(V) :- end(T), lineOnAns(T,V).
ans(V) :- end(T), existAns(T,V).
ans(V) :- end(T), stationsAns(T,V).

#show ans/1.
% Added rules to handle new instances
