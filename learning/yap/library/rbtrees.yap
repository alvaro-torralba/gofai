/**
 * @file   rbtrees.yap
 * @author VITOR SANTOS COSTA <vsc@VITORs-MBP.lan>
 * @author Jan Wielemaker
 * @date   Wed Nov 18 00:11:41 2015
 *
 * @brief  Red-Black trees
 *
 *
*/


:- module(rbtrees,
	  [rb_new/1,
	   rb_empty/1,		% ?T
	   rb_lookup/3,		% +Key, -Value, +T
	   rb_update/4,		% +T, +Key, +NewVal, -TN
	   rb_update/5,		% +T, +Key, ?OldVal, +NewVal, -TN
	   rb_rewrite/3,		% +T, +Key, +NewVal
	   rb_rewrite/4,		% +T, +Key, ?OldVal, +NewVal
	   rb_apply/4,			% +T, +Key, :G, -TN
	   rb_lookupall/3,		% +Key, -Value, +T
	   rb_insert/4,		% +T0, +Key, ?Value, -TN
	   rb_insert_new/4,	% +T0, +Key, ?Value, -TN
	   rb_delete/3,		% +T, +Key, -TN
	   rb_delete/4,		% +T, +Key, -Val, -TN
	   rb_visit/2,			% +T, -Pairs
	   rb_visit/3,
	   rb_keys/2,			% +T, +Keys
	   rb_keys/3,
	   rb_map/2,
	   rb_map/3,
	   rb_partial_map/4,
	   rb_clone/4,
	   rb_min/3,
	   rb_max/3,
	   rb_del_min/4,
	   rb_del_max/4,
	   rb_next/4,
	   rb_previous/4,
	   rb_fold/4,
	   rb_key_fold/4,
	   list_to_rbtree/2,
	   ord_list_to_rbtree/2,
	   keys_to_rbtree/2,
	   ord_keys_to_rbtree/2,
	   is_rbtree/1,
	   rb_size/2,
	   rb_in/3
       ]).

/**
  *
  * @defgroup rbtrees Red-Black Trees
  * @ingroup YAPLibrary
  * @{
  *

Red-Black trees are balanced search binary trees. They are named because
nodes can be classified as either red or   black. The code we include is
based on "Introduction  to  Algorithms",   second  edition,  by  Cormen,
Leiserson, Rivest and Stein. The library   includes  routines to insert,
lookup and delete elements in the tree.

A Red black tree is represented as a term t(Nil, Tree), where Nil is the
Nil-node, a node shared for each nil-node in  the tree. Any node has the
form colour(Left, Key, Value, Right), where _colour_  is one of =red= or
=black=.

@author Vitor Santos Costa, Jan Wielemaker
*/

:- meta_predicate rb_map(+,2,-),
	rb_partial_map(+,+,2,-),
	rb_apply(+,+,2,-).

:- use_module(library(maplist)).

/*
:- use_module(library(type_check)).

:- type rbtree(K,V) ---> t(tree(K,V),tree(K,V)).
:- type tree(K,V)   ---> black(tree(K,V),K,V,tree(K,V))
	               ; red(tree(K,V),K,V,tree(K,V))
		       ; ''.
:- type cmp ---> (=) ; (<) ; (>).


:- pred rb_new(rbtree(_K,_V)).
:- pred rb_empty(rbtree(_K,_V)).
:- pred rb_lookup(K,V,rbtree(K,V)).
:- pred lookup(K,V, tree(K,V)).
:- pred lookup(cmp, K, V, tree(K,V)).
:- pred rb_min(rbtree(K,V),K,V).
:- pred min(tree(K,V),K,V).
:- pred rb_max(rbtree(K,V),K,V).
:- pred max(tree(K,V),K,V).
:- pred rb_next(rbtree(K,V),K,pair(K,V),V).
:- pred next(tree(K,V),K,pair(K,V),V,tree(K,V)).
*/

/** @pred rb_new(? _T_)


Create a new tree.

	@deprecated	Use rb_empty/1.
*/
rb_new(t(Nil,Nil)) :- Nil = black('',_,_,'').

rb_new(K,V,t(Nil,black(Nil,K,V,Nil))) :- Nil = black('',_,_,'').

/** @pred rb_empty(? _T_)


Succeeds if tree  _T_ is empty.


*/rb_empty(t(Nil,Nil)) :- Nil = black('',_,_,'').

/** @pred rb_lookup(+ _Key_,- _Value_,+ _T_)


Backtrack through all elements with key  _Key_ in the red-black tree
 _T_, returning for each the value  _Value_.


*/
rb_lookup(Key, Val, t(_,Tree)) :-
	lookup(Key, Val, Tree).

lookup(_, _, black('',_,_,'')) :- !, fail.
lookup(Key, Val, Tree) :-
	arg(2,Tree,KA),
	compare(Cmp,KA,Key),
	lookup(Cmp,Key,Val,Tree).

lookup(>, K, V, Tree) :-
	arg(1,Tree,NTree),
	lookup(K, V, NTree).
lookup(<, K, V, Tree) :-
	arg(4,Tree,NTree),
	lookup(K, V, NTree).
lookup(=, _, V, Tree) :-
	arg(3,Tree,V).

/** @pred rb_min(+ _T_,- _Key_,- _Value_)


 _Key_  is the minimum key in  _T_, and is associated with  _Val_.


*/
rb_min(t(_,Tree), Key, Val) :-
	min(Tree, Key, Val).

min(red(black('',_,_,_),Key,Val,_), Key, Val) :- !.
min(black(black('',_,_,_),Key,Val,_), Key, Val) :- !.
min(red(Right,_,_,_), Key, Val) :-
	min(Right,Key,Val).
min(black(Right,_,_,_), Key, Val) :-
	min(Right,Key,Val).

/** @pred rb_max(+ _T_,- _Key_,- _Value_)


 _Key_  is the maximal key in  _T_, and is associated with  _Val_.


*/
rb_max(t(_,Tree), Key, Val) :-
	max(Tree, Key, Val).

max(red(_,Key,Val,black('',_,_,_)), Key, Val) :- !.
max(black(_,Key,Val,black('',_,_,_)), Key, Val) :- !.
max(red(_,_,_,Left), Key, Val) :-
	max(Left,Key,Val).
max(black(_,_,_,Left), Key, Val) :-
	max(Left,Key,Val).

/** @pred rb_next(+ _T_, + _Key_,- _Next_,- _Value_)


 _Next_ is the next element after  _Key_ in  _T_, and is
associated with  _Val_.


*/
rb_next(t(_,Tree), Key, Next, Val) :-
	next(Tree, Key, Next, Val, []).

next(black('',_,_,''), _, _, _, _) :- !, fail.
next(Tree, Key, Next, Val, Candidate) :-
	arg(2,Tree,KA),
	arg(3,Tree,VA),
	compare(Cmp,KA,Key),
	next(Cmp, Key, KA, VA, Next, Val, Tree, Candidate).

next(>, K, KA, VA, NK, V, Tree, _) :-
	arg(1,Tree,NTree),
	next(NTree,K,NK,V,KA-VA).
next(<, K, _, _, NK, V, Tree, Candidate) :-
	arg(4,Tree,NTree),
	next(NTree,K,NK,V,Candidate).
next(=, _, _, _, NK, Val, Tree, Candidate) :-
	arg(4,Tree,NTree),
	(
	    min(NTree, NK, Val)
	-> true
	;
	    Candidate = (NK-Val)
	).

%%	@pred rb_previous(+T, +Key, -Previous, -Value) is semidet.
%
%	Previous is the  previous  element  after   Key  in  T,  and  is
%	associated with Val.

rb_previous(t(_,Tree), Key, Previous, Val) :-
	previous(Tree, Key, Previous, Val, []).

previous(black('',_,_,''), _, _, _, _) :- !, fail.
previous(Tree, Key, Previous, Val, Candidate) :-
	arg(2,Tree,KA),
	arg(3,Tree,VA),
	compare(Cmp,KA,Key),
	previous(Cmp, Key, KA, VA, Previous, Val, Tree, Candidate).

previous(>, K, _, _, NK, V, Tree, Candidate) :-
	arg(1,Tree,NTree),
	previous(NTree,K,NK,V,Candidate).
previous(<, K, KA, VA, NK, V, Tree, _) :-
	arg(4,Tree,NTree),
	previous(NTree,K,NK,V,KA-VA).
previous(=, _, _, _, K, Val, Tree, Candidate) :-
	arg(1,Tree,NTree),
	(
	    max(NTree, K, Val)
	-> true
	;
	    Candidate = (K-Val)
	).

/** @pred rb_update(+ _T_,+ _Key_,+ _NewVal_,- _TN_)


Tree  _TN_ is tree  _T_, but with value for  _Key_ associated
with  _NewVal_. Fails if it cannot find  _Key_ in  _T_.


*/
rb_update(t(Nil,OldTree), Key, OldVal, Val, t(Nil,NewTree)) :-
	update(OldTree, Key, OldVal, Val, NewTree).

/** @pred rb_update(+ _T_,+ _Key_,- _OldVal_,+ _NewVal_,- _TN_)


Tree  _TN_ is tree  _T_, but with value for  _Key_ associated
with  _NewVal_. Fails if it cannot find  _Key_ in  _T_.


*/
rb_update(t(Nil,OldTree), Key, Val, t(Nil,NewTree)) :-
	update(OldTree, Key, _, Val, NewTree).

update(black(Left,Key0,Val0,Right), Key, OldVal, Val, NewTree) :-
	Left \= [],
	compare(Cmp,Key0,Key),
	(Cmp == (=)
	-> OldVal = Val0,
	    NewTree = black(Left,Key0,Val,Right)
	;
	Cmp == (>) ->
	   NewTree = black(NewLeft,Key0,Val0,Right),
	  update(Left, Key, OldVal, Val, NewLeft)
	;
	  NewTree = black(Left,Key0,Val0,NewRight),
	  update(Right, Key, OldVal, Val, NewRight)
	).
update(red(Left,Key0,Val0,Right), Key, OldVal, Val, NewTree) :-
	compare(Cmp,Key0,Key),
	(Cmp == (=)
	-> OldVal = Val0,
	    NewTree = red(Left,Key0,Val,Right)
	;
	Cmp == (>)
	-> NewTree = red(NewLeft,Key0,Val0,Right),
	  update(Left, Key, OldVal, Val, NewLeft)
	;
	  NewTree = red(Left,Key0,Val0,NewRight),
	  update(Right, Key, OldVal, Val, NewRight)
	).

/**	@pred rb_rewrite(+T, +Key, +NewVal) is semidet.

	Tree T has   value  for  Key  associated with
	NewVal.  Fails if it cannot find Key in T.
*/
/**	@pred rb_rewrite(+T, +Key, ?OldVal, +NewVal) is semidet.
 
	Tree T has   value  for  Key  associated with
	NewVal.  Fails if it cannot find Key in T.
*/
rb_rewrite(t(_Nil,OldTree), Key, OldVal, Val) :-
	rewrite(OldTree, Key, OldVal, Val).

rb_rewrite(t(_Nil,OldTree), Key, Val) :-
	rewrite(OldTree, Key, _, Val).

rewrite(Node, Key, OldVal, Val) :-
	Node = black(Left,Key0,Val0,Right),
	Left \= [],
	compare(Cmp,Key0,Key),
	(Cmp == (=)
	-> OldVal = Val0,
	    setarg(3, Node, Val)
	;
	Cmp == (>) ->
	  rewrite(Left, Key, OldVal, Val)
	;
	  rewrite(Right, Key, OldVal, Val)
	).
 rewrite(Node, Key, OldVal, Val) :-
	Node = red(Left,Key0,Val0,Right),
	Left \= [],
	compare(Cmp,Key0,Key),
	(
           Cmp == (=)
	->
            OldVal = Val0,
	    setarg(3, Node, Val)
	;
	  Cmp == (>)
         ->
	  rewrite(Left, Key, OldVal, Val)
	;
	  rewrite(Right, Key, OldVal, Val)
	).


/** @pred rb_apply(+ _T_,+ _Key_,+ _G_,- _TN_)


  If the value associated with key  _Key_ is  _Val0_ in  _T_, and
if `call(G,Val0,ValF)` holds, then  _TN_ differs from
 _T_ only in that  _Key_ is associated with value  _ValF_ in
tree  _TN_. Fails if it cannot find  _Key_ in  _T_, or if
`call(G,Val0,ValF)` is not satisfiable.


*/
rb_apply(t(Nil,OldTree), Key, Goal, t(Nil,NewTree)) :-
	apply(OldTree, Key, Goal, NewTree).

%apply(black('',_,_,''), _, _, _) :- !, fail.
apply(black(Left,Key0,Val0,Right), Key, Goal,
      black(NewLeft,Key0,Val,NewRight)) :-
	Left \= [],
	compare(Cmp,Key0,Key),
	(Cmp == (=)
	-> NewLeft = Left,
	    NewRight = Right,
	    call(Goal,Val0,Val)
	; Cmp == (>)
	->  NewRight = Right,
	    Val = Val0,
	    apply(Left, Key, Goal, NewLeft)
	;
	    NewLeft = Left,
	    Val = Val0,
	    apply(Right, Key, Goal, NewRight)
	).
apply(red(Left,Key0,Val0,Right), Key, Goal,
      red(NewLeft,Key0,Val,NewRight)) :-
	compare(Cmp,Key0,Key),
	( Cmp == (=)
	-> NewLeft = Left,
	 NewRight = Right,
	 call(Goal,Val0,Val)
	; Cmp == (>)
	-> NewRight = Right,
	 Val = Val0,
	 apply(Left, Key, Goal, NewLeft)
	;
	 NewLeft = Left,
	 Val = Val0,
	 apply(Right, Key, Goal, NewRight)
	).

%%	rb_in(?Key, ?Val, +Tree) is nondet.
%
%	True if Key-Val appear in Tree. Uses indexing if Key is bound.

rb_in(Key, Val, t(_,T)) :-
	var(Key), !,
	enum(Key, Val, T).
rb_in(Key, Val, t(_,T)) :-
	lookup(Key, Val, T).


enum(Key, Val, black(L,K,V,R)) :-
	L \= '',
	enum_cases(Key, Val, L, K, V, R).
enum(Key, Val, red(L,K,V,R)) :-
	enum_cases(Key, Val, L, K, V, R).

enum_cases(Key, Val, L, _, _, _) :-
	enum(Key, Val, L).
enum_cases(Key, Val, _, Key, Val, _).
enum_cases(Key, Val, _, _, _, R) :-
	enum(Key, Val, R).


/** @pred rb_lookupall(+ _Key_,- _Value_,+ _T_)


Lookup all elements with key  _Key_ in the red-black tree
 _T_, returning the value  _Value_.


*/
rb_lookupall(Key, Val, t(_,Tree)) :-
	lookupall(Key, Val, Tree).


lookupall(_, _, black('',_,_,'')) :- !, fail.
lookupall(Key, Val, Tree) :-
	arg(2,Tree,KA),
	compare(Cmp,KA,Key),
	lookupall(Cmp,Key,Val,Tree).

lookupall(>, K, V, Tree) :-
	arg(4,Tree,NTree),
	rb_lookupall(K, V, NTree).
lookupall(=, _, V, Tree) :-
	arg(3,Tree,V).
lookupall(=, K, V, Tree) :-
	arg(1,Tree,NTree),
	lookupall(K, V, NTree).
lookupall(<, K, V, Tree) :-
	arg(1,Tree,NTree),
	lookupall(K, V, NTree).

		 /*******************************
		 *	 TREE INSERTION		*
		 *******************************/

/** @pred rb_insert(+ _T0_,+ _Key_,? _Value_,+ _TF_)


Add an element with key  _Key_ and  _Value_ to the tree
 _T0_ creating a new red-black tree  _TF_. Duplicated elements are not
allowed.

Add a new element with key  _Key_ and  _Value_ to the tree
 _T0_ creating a new red-black tree  _TF_. Fails is an element
with  _Key_ exists in the tree.


*/
rb_insert(t(Nil,Tree0),Key,Val,t(Nil,Tree)) :-
	insert(Tree0,Key,Val,Nil,Tree).


insert(Tree0,Key,Val,Nil,Tree) :-
	insert2(Tree0,Key,Val,Nil,TreeI,_),
	fix_root(TreeI,Tree).

%
% Cormen et al present the algorithm as
% (1) standard tree insertion;
% (2) from the viewpoint of the newly inserted node:
%     partially fix the tree;
%     move upwards
% until reaching the root.
%
% We do it a little bit different:
%
% (1) standard tree insertion;
% (2) move upwards:
%      when reaching a black node;
%        if the tree below may be broken, fix it.
% We take advantage of Prolog unification
% to do several operations in a single go.
%



%
% actual insertion
%
insert2(black('',_,_,''), K, V, Nil, T, Status) :- !,
	T = red(Nil,K,V,Nil),
	Status = not_done.
insert2(red(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	( K @< K0
	-> NR = R,
	  NT = red(NL,K0,V0,R),
	  insert2(L, K, V, Nil, NL, Flag)
	; K == K0 ->
	  NT = red(L,K0,V,R),
	  Flag = done
	;
	  NT = red(L,K0,V0,NR),
	  insert2(R, K, V, Nil, NR, Flag)
	).
insert2(black(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	( K @< K0
	-> insert2(L, K, V, Nil, IL, Flag0),
	  fix_left(Flag0, black(IL,K0,V0,R), NT, Flag)
	; K == K0 ->
	  NT = 	black(L,K0,V,R),
	  Flag = done
	;
	  insert2(R, K, V, Nil, IR, Flag0),
	  fix_right(Flag0, black(L,K0,V0,IR), NT, Flag)
	).

% We don't use parent nodes, so we may have to fix the root.

%%	rb_insert_new(+T0, +Key, ?Value, -TN) is semidet.
%
%	Add a new element with key Key and Value  to the tree T0 creating a
%	new red-black tree TN.   Fails if Key is a key in T0.

rb_insert_new(t(Nil,Tree0),Key,Val,t(Nil,Tree)) :-
	insert_new(Tree0,Key,Val,Nil,Tree).


insert_new(Tree0,Key,Val,Nil,Tree) :-
	insert_new_2(Tree0,Key,Val,Nil,TreeI,_),
	fix_root(TreeI,Tree).

%
% actual insertion, copied from insert2
%
insert_new_2(black('',_,_,''), K, V, Nil, T, Status) :- !,
	T = red(Nil,K,V,Nil),
	Status = not_done.
insert_new_2(red(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	( K @< K0
	-> NR = R,
	  NT = red(NL,K0,V0,R),
	  insert_new_2(L, K, V, Nil, NL, Flag)
	; K == K0 ->
	  fail
	;
	  NT = red(L,K0,V0,NR),
	  insert_new_2(R, K, V, Nil, NR, Flag)
	).
insert_new_2(black(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	( K @< K0
	-> insert_new_2(L, K, V, Nil, IL, Flag0),
	  fix_left(Flag0, black(IL,K0,V0,R), NT, Flag)
	; K == K0 ->
	  fail
	;
	  insert_new_2(R, K, V, Nil, IR, Flag0),
	  fix_right(Flag0, black(L,K0,V0,IR), NT, Flag)
	).

%
% make sure the root is always black.
%
fix_root(black(L,K,V,R),black(L,K,V,R)).
fix_root(red(L,K,V,R),black(L,K,V,R)).



%
% How to fix if we have inserted on the left
%
fix_left(done,T,T,done) :- !.
fix_left(not_done,Tmp,Final,Done) :-
	fix_left(Tmp,Final,Done).

%
% case 1 of RB: just need to change colors.
%
fix_left(black(red(Al,AK,AV,red(Be,BK,BV,Ga)),KC,VC,red(De,KD,VD,Ep)),
	red(black(Al,AK,AV,red(Be,BK,BV,Ga)),KC,VC,black(De,KD,VD,Ep)),
	not_done) :- !.
fix_left(black(red(red(Al,KA,VA,Be),KB,VB,Ga),KC,VC,red(De,KD,VD,Ep)),
	red(black(red(Al,KA,VA,Be),KB,VB,Ga),KC,VC,black(De,KD,VD,Ep)),
	not_done) :- !.
%
% case 2 of RB: got a knee so need to do rotations
%
fix_left(black(red(Al,KA,VA,red(Be,KB,VB,Ga)),KC,VC,De),
	black(red(Al,KA,VA,Be),KB,VB,red(Ga,KC,VC,De)),
	done) :- !.
%
% case 3 of RB: got a line
%
fix_left(black(red(red(Al,KA,VA,Be),KB,VB,Ga),KC,VC,De),
	black(red(Al,KA,VA,Be),KB,VB,red(Ga,KC,VC,De)),
	done) :- !.
%
% case 4 of RB: nothing to do
%
fix_left(T,T,done).

%
% How to fix if we have inserted on the right
%
fix_right(done,T,T,done) :- !.
fix_right(not_done,Tmp,Final,Done) :-
	fix_right(Tmp,Final,Done).

%
% case 1 of RB: just need to change colors.
%
fix_right(black(red(Ep,KD,VD,De),KC,VC,red(red(Ga,KB,VB,Be),KA,VA,Al)),
	red(black(Ep,KD,VD,De),KC,VC,black(red(Ga,KB,VB,Be),KA,VA,Al)),
	not_done) :- !.
fix_right(black(red(Ep,KD,VD,De),KC,VC,red(Ga,Ka,Va,red(Be,KB,VB,Al))),
	red(black(Ep,KD,VD,De),KC,VC,black(Ga,Ka,Va,red(Be,KB,VB,Al))),
	not_done) :- !.
%
% case 2 of RB: got a knee so need to do rotations
%
fix_right(black(De,KC,VC,red(red(Ga,KB,VB,Be),KA,VA,Al)),
	black(red(De,KC,VC,Ga),KB,VB,red(Be,KA,VA,Al)),
	done) :- !.
%
% case 3 of RB: got a line
%
fix_right(black(De,KC,VC,red(Ga,KB,VB,red(Be,KA,VA,Al))),
	black(red(De,KC,VC,Ga),KB,VB,red(Be,KA,VA,Al)),
	done) :- !.
%
% case 4 of RB: nothing to do.
%
fix_right(T,T,done).

%
% simplified processor
%
%
pretty_print(t(_,T)) :-
	pretty_print(T,6).

pretty_print(black('',_,_,''),_) :- !.
pretty_print(red(L,K,_,R),D) :-
	DN is D+6,
	pretty_print(L,DN),
	format("~t~a:~d~*|~n",[r,K,D]),
	pretty_print(R,DN).
pretty_print(black(L,K,_,R),D) :-
	DN is D+6,
	pretty_print(L,DN),
	format("~t~a:~d~*|~n",[b,K,D]),
	pretty_print(R,DN).


/** @pred rb_delete(+ _T_,+ _Key_,- _TN_)


Delete element with key  _Key_ from the tree  _T_, returning a new
tree  _TN_.


*/
rb_delete(t(Nil,T), K, t(Nil,NT)) :-
	delete(T, K, _, NT, _).

/** @pred rb_delete(+ _T_,+ _Key_,- _Val_,- _TN_)

Delete element with key  _Key_ from the tree  _T_, returning the
value  _Val_ associated with the key and a new tree  _TN_.


*/
rb_delete(t(Nil,T), K, V, t(Nil,NT)) :-
	delete(T, K, V0, NT, _),
	V = V0.

%
% I am afraid our representation is not as nice for delete
%
delete(red(L,K0,V0,R), K, V, NT, Flag) :-
	K @< K0, !,
	delete(L, K, V, NL, Flag0),
	fixup_left(Flag0,red(NL,K0,V0,R),NT, Flag).
delete(red(L,K0,V0,R), K, V, NT, Flag) :-
	K @> K0, !,
	delete(R, K, V, NR, Flag0),
	fixup_right(Flag0,red(L,K0,V0,NR),NT, Flag).
delete(red(L,_,V,R), _, V, OUT, Flag) :-
%	K == K0,
	delete_red_node(L,R,OUT,Flag).
delete(black(L,K0,V0,R), K, V, NT, Flag) :-
	K @< K0, !,
	delete(L, K, V, NL, Flag0),
	fixup_left(Flag0,black(NL,K0,V0,R),NT, Flag).
delete(black(L,K0,V0,R), K, V, NT, Flag) :-
	K @> K0, !,
	delete(R, K, V, NR, Flag0),
	fixup_right(Flag0,black(L,K0,V0,NR),NT, Flag).
delete(black(L,_,V,R), _, V, OUT, Flag) :-
%	K == K0,
	delete_black_node(L,R,OUT,Flag).

/** @pred rb_del_min(+ _T_,- _Key_,- _Val_,- _TN_)


Delete the least element from the tree  _T_, returning the key
 _Key_, the value  _Val_ associated with the key and a new tree
 _TN_.


*/
rb_del_min(t(Nil,T), K, Val, t(Nil,NT)) :-
	del_min(T, K, Val, Nil, NT, _).

del_min(red(black('',_,_,_),K,V,R), K, V, Nil, OUT, Flag) :- !,
	delete_red_node(Nil,R,OUT,Flag).
del_min(red(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	del_min(L, K, V, Nil, NL, Flag0),
	fixup_left(Flag0,red(NL,K0,V0,R), NT, Flag).
del_min(black(black('',_,_,_),K,V,R), K, V, Nil, OUT, Flag) :- !,
	delete_black_node(Nil,R,OUT,Flag).
del_min(black(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	del_min(L, K, V, Nil, NL, Flag0),
	fixup_left(Flag0,black(NL,K0,V0,R),NT, Flag).


/** @pred rb_del_max(+ _T_,- _Key_,- _Val_,- _TN_)


Delete the largest element from the tree  _T_, returning the key
 _Key_, the value  _Val_ associated with the key and a new tree
 _TN_.


*/
rb_del_max(t(Nil,T), K, Val, t(Nil,NT)) :-
	del_max(T, K, Val, Nil, NT, _).

del_max(red(L,K,V,black('',_,_,_)), K, V, Nil, OUT, Flag) :- !,
	delete_red_node(L,Nil,OUT,Flag).
del_max(red(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	del_max(R, K, V, Nil, NR, Flag0),
	fixup_right(Flag0,red(L,K0,V0,NR),NT, Flag).
del_max(black(L,K,V,black('',_,_,_)), K, V, Nil, OUT, Flag) :- !,
	delete_black_node(L,Nil,OUT,Flag).
del_max(black(L,K0,V0,R), K, V, Nil, NT, Flag) :-
	del_max(R, K, V, Nil, NR, Flag0),
	fixup_right(Flag0,black(L,K0,V0,NR), NT, Flag).



delete_red_node(L1,L2,L1,done) :- L1 == L2, !.
delete_red_node(black('',_,_,''),R,R,done) :-  !.
delete_red_node(L,black('',_,_,''),L,done) :-  !.
delete_red_node(L,R,OUT,Done) :-
	delete_next(R,NK,NV,NR,Done0),
	fixup_right(Done0,red(L,NK,NV,NR),OUT,Done).


delete_black_node(L1,L2,L1,not_done) :- 	L1 == L2, !.
delete_black_node(black('',_,_,''),red(L,K,V,R),black(L,K,V,R),done) :- !.
delete_black_node(black('',_,_,''),R,R,not_done) :- !.
delete_black_node(red(L,K,V,R),black('',_,_,''),black(L,K,V,R),done) :- !.
delete_black_node(L,black('',_,_,''),L,not_done) :- !.
delete_black_node(L,R,OUT,Done) :-
	delete_next(R,NK,NV,NR,Done0),
	fixup_right(Done0,black(L,NK,NV,NR),OUT,Done).


delete_next(red(black('',_,_,''),K,V,R),K,V,R,done) :- 	!.
delete_next(black(black('',_,_,''),K,V,red(L1,K1,V1,R1)),
	K,V,black(L1,K1,V1,R1),done) :- !.
delete_next(black(black('',_,_,''),K,V,R),K,V,R,not_done) :- !.
delete_next(red(L,K,V,R),K0,V0,OUT,Done) :-
	delete_next(L,K0,V0,NL,Done0),
	fixup_left(Done0,red(NL,K,V,R),OUT,Done).
delete_next(black(L,K,V,R),K0,V0,OUT,Done) :-
	delete_next(L,K0,V0,NL,Done0),
	fixup_left(Done0,black(NL,K,V,R),OUT,Done).


fixup_left(done,T,T,done).
fixup_left(not_done,T,NT,Done) :-
	fixup2(T,NT,Done).


%
% case 1: x moves down, so we have to try to fix it again.
% case 1 -> 2,3,4 -> done
%
fixup2(black(black(Al,KA,VA,Be),KB,VB,red(black(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),
	black(T1,KD,VD,black(Ep,KE,VE,Fi)),done) :- !,
	fixup2(red(black(Al,KA,VA,Be),KB,VB,black(Ga,KC,VC,De)),
		T1,
                _).
%
% case 2: x moves up, change one to red
%
fixup2(red(black(Al,KA,VA,Be),KB,VB,black(black(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),
	black(black(Al,KA,VA,Be),KB,VB,red(black(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),done) :- !.
fixup2(black(black(Al,KA,VA,Be),KB,VB,black(black(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),
	black(black(Al,KA,VA,Be),KB,VB,red(black(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),not_done) :- !.
%
% case 3: x stays put, shift left and do a 4
%
fixup2(red(black(Al,KA,VA,Be),KB,VB,black(red(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),
	red(black(black(Al,KA,VA,Be),KB,VB,Ga),KC,VC,black(De,KD,VD,black(Ep,KE,VE,Fi))),
	done) :- !.
fixup2(black(black(Al,KA,VA,Be),KB,VB,black(red(Ga,KC,VC,De),KD,VD,black(Ep,KE,VE,Fi))),
	black(black(black(Al,KA,VA,Be),KB,VB,Ga),KC,VC,black(De,KD,VD,black(Ep,KE,VE,Fi))),
	done) :- !.
%
% case 4: rotate left, get rid of red
%
fixup2(red(black(Al,KA,VA,Be),KB,VB,black(C,KD,VD,red(Ep,KE,VE,Fi))),
	red(black(black(Al,KA,VA,Be),KB,VB,C),KD,VD,black(Ep,KE,VE,Fi)),
	done).
fixup2(black(black(Al,KA,VA,Be),KB,VB,black(C,KD,VD,red(Ep,KE,VE,Fi))),
	black(black(black(Al,KA,VA,Be),KB,VB,C),KD,VD,black(Ep,KE,VE,Fi)),
	done).


fixup_right(done,T,T,done).
fixup_right(not_done,T,NT,Done) :-
	fixup3(T,NT,Done).



%
% case 1: x moves down, so we have to try to fix it again.
% case 1 -> 2,3,4 -> done
%
fixup3(black(red(black(Fi,KE,VE,Ep),KD,VD,black(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	black(black(Fi,KE,VE,Ep),KD,VD,T1),done) :- !,
        fixup3(red(black(De,KC,VC,Ga),KB,VB,black(Be,KA,VA,Al)),T1,_).

%
% case 2: x moves up, change one to red
%
fixup3(red(black(black(Fi,KE,VE,Ep),KD,VD,black(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	black(red(black(Fi,KE,VE,Ep),KD,VD,black(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	done) :- !.
fixup3(black(black(black(Fi,KE,VE,Ep),KD,VD,black(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	black(red(black(Fi,KE,VE,Ep),KD,VD,black(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	not_done):- !.
%
% case 3: x stays put, shift left and do a 4
%
fixup3(red(black(black(Fi,KE,VE,Ep),KD,VD,red(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	red(black(black(Fi,KE,VE,Ep),KD,VD,De),KC,VC,black(Ga,KB,VB,black(Be,KA,VA,Al))),
	done) :- !.
fixup3(black(black(black(Fi,KE,VE,Ep),KD,VD,red(De,KC,VC,Ga)),KB,VB,black(Be,KA,VA,Al)),
	black(black(black(Fi,KE,VE,Ep),KD,VD,De),KC,VC,black(Ga,KB,VB,black(Be,KA,VA,Al))),
	done) :- !.
%
% case 4: rotate right, get rid of red
%
fixup3(red(black(red(Fi,KE,VE,Ep),KD,VD,C),KB,VB,black(Be,KA,VA,Al)),
	red(black(Fi,KE,VE,Ep),KD,VD,black(C,KB,VB,black(Be,KA,VA,Al))),
	done).
fixup3(black(black(red(Fi,KE,VE,Ep),KD,VD,C),KB,VB,black(Be,KA,VA,Al)),
	black(black(Fi,KE,VE,Ep),KD,VD,black(C,KB,VB,black(Be,KA,VA,Al))),
	done).


%
% whole list
%

/** @pred rb_visit(+ _T_,- _Pairs_)


 _Pairs_ is an infix visit of tree  _T_, where each element of
 _Pairs_ is of the form   _K_- _Val_.

*/
rb_visit(t(_,T),Lf) :-
	visit(T,[],Lf).

rb_visit(t(_,T),L0,Lf) :-
	visit(T,L0,Lf).

visit(black('',_,_,_),L,L) :- !.
visit(red(L,K,V,R),L0,Lf) :-
	visit(L,[K-V|L1],Lf),
	visit(R,L0,L1).
visit(black(L,K,V,R),L0,Lf) :-
	visit(L,[K-V|L1],Lf),
	visit(R,L0,L1).

:- meta_predicate map(?,2,?,?).  % this is required.

/** @pred rb_map(+ _T_,+ _G_,- _TN_)


For all nodes  _Key_ in the tree  _T_, if the value associated with
key  _Key_ is  _Val0_ in tree  _T_, and if
`call(G,Val0,ValF)` holds, then the value associated with  _Key_
in  _TN_ is  _ValF_. Fails if or if `call(G,Val0,ValF)` is not
satisfiable for all  _Var0_.


*/
rb_map(t(Nil,Tree),Goal,t(Nil,NewTree)) :-
	map(Tree,Goal,NewTree,Nil).


map(black('',_,_,''),_,Nil,Nil) :- !.
map(red(L,K,V,R),Goal,red(NL,K,NV,NR),Nil) :-
	call(Goal,V,NV), !,
	map(L,Goal,NL,Nil),
	map(R,Goal,NR,Nil).
map(black(L,K,V,R),Goal,black(NL,K,NV,NR),Nil) :-
	call(Goal,V,NV), !,
	map(L,Goal,NL,Nil),
	map(R,Goal,NR,Nil).

:- meta_predicate rb_map(?,1). % this is not strictly required
:- meta_predicate map(?,1).  % this is required.

%%	rb_map(+T, :G, -TN) is semidet.
%
%	For all nodes Key in the tree   T,  if the value associated with
%	key Key is Val0 in tree T,  and if call(G,Val0,ValF) holds, then
%	the  value  associated  with  Key  in   TN  is  ValF.  Fails  if
%	call(G,Val0,ValF) is not satisfiable for all Var0.

rb_map(t(_,Tree),Goal) :-
	map(Tree,Goal).


map(black('',_,_,''),_) :- !.
map(red(L,_,V,R),Goal) :-
	call(Goal,V), !,
	map(L,Goal),
	map(R,Goal).
map(black(L,_,V,R),Goal) :-
	call(Goal,V), !,
	map(L,Goal),
	map(R,Goal).

:- meta_predicate rb_fold(3,?,?,?).  % this is required.
:- meta_predicate map_acc(?,3,?,?).  % this is required.

/** @pred rb_fold(+ _T_,+ _G_,+ _Acc0_, - _AccF_)


For all nodes  _Key_ in the tree  _T_, if the value
associated with key  _Key_ is  _V_ in tree  _T_, if
`call(G,V,Acc1,Acc2)` holds, then if  _VL_ is value of the
previous node in inorder, `call(G,VL,_,Acc0)` must hold, and if
 _VR_ is the value of the next node in inorder,
`call(G,VR,Acc1,_)` must hold.


*/rb_fold(Goal, t(_,Tree), In, Out) :-
	map_acc(Tree, Goal, In, Out).

map_acc(black('',_,_,''), _, Acc, Acc) :- !.
map_acc(red(L,_,V,R), Goal, Left, Right) :-
	map_acc(L,Goal, Left, Left1),
	once(call(Goal,V, Left1, Right1)),
	map_acc(R,Goal, Right1, Right).
map_acc(black(L,_,V,R), Goal, Left, Right) :-
	map_acc(L,Goal, Left, Left1),
	once(call(Goal,V, Left1, Right1)),
	map_acc(R,Goal, Right1, Right).

:- meta_predicate rb_key_fold(4,?,?,?).  % this is required.
:- meta_predicate map_key_acc(?,4,?,?).  % this is required.

/** @pred rb_key_fold(+ _T_,+ _G_,+ _Acc0_, - _AccF_)


For all nodes  _Key_ in the tree  _T_, if the value
associated with key  _Key_ is  _V_ in tree  _T_, if
`call(G,Key,V,Acc1,Acc2)` holds, then if  _VL_ is value of the
previous node in inorder, `call(G,KeyL,VL,_,Acc0)` must hold, and if
 _VR_ is the value of the next node in inorder,
`call(G,KeyR,VR,Acc1,_)` must hold.


*/
rb_key_fold(Goal, t(_,Tree), In, Out) :-
	map_key_acc(Tree, Goal, In, Out).

map_key_acc(black('',_,_,''), _, Acc, Acc) :- !.
map_key_acc(red(L,Key,V,R), Goal, Left, Right) :-
	map_key_acc(L,Goal, Left, Left1),
	once(call(Goal, Key, V, Left1, Right1)),
	map_key_acc(R,Goal, Right1, Right).
map_key_acc(black(L,Key,V,R), Goal, Left, Right) :-
	map_key_acc(L,Goal, Left, Left1),
	once(call(Goal, Key, V, Left1, Right1)),
	map_key_acc(R,Goal, Right1, Right).

/** @pred rb_clone(+ _T_,?_OldNodes_,?_NT_,?_Nodes_)


=Clone= the red-back tree into a new tree with the same keys as the
original but with all values set to unbound values. _Nodes_ is a list
containing all new nodes as pairs  _K-V_.


*/

rb_clone(t(Nil,T),ONs,t(Nil,NT),Ns) :-
	clone(T,Nil,ONs,[],NT,Ns,[]).

clone(black('',_,_,''),Nil,ONs,ONs,Nil,Ns,Ns) :- !.
clone(red(L,K,V,R),Nil,ONsF,ONs0,red(NL,K,NV,NR),NsF,Ns0) :-
	clone(L,Nil,ONsF,[K-V|ONs1],NL,NsF,[K-NV|Ns1]),
	clone(R,Nil,ONs1,ONs0,NR,Ns1,Ns0).
clone(black(L,K,V,R),Nil,ONsF,ONs0,black(NL,K,NV,NR),NsF,Ns0) :-
	clone(L,Nil,ONsF,[K-V|ONs1],NL,NsF,[K-NV|Ns1]),
	clone(R,Nil,ONs1,ONs0,NR,Ns1,Ns0).

/** @pred rb_partial_map(+ _T_,+ _Keys_,+ _G_,- _TN_)


For all nodes  _Key_ in  _Keys_, if the value associated with key
 _Key_ is  _Val0_ in tree  _T_, and if `call(G,Val0,ValF)`
holds, then the value associated with  _Key_ in  _TN_ is
 _ValF_. Fails if or if `call(G,Val0,ValF)` is not satisfiable
for all  _Var0_. Assumes keys are not repeated.


*/
rb_partial_map(t(Nil,T0), Map, Goal, t(Nil,TF)) :-
	partial_map(T0, Map, [], Nil, Goal, TF).

rb_partial_map(t(Nil,T0), Map, Map0, Goal, t(Nil,TF)) :-
	partial_map(T0, Map, Map0, Nil, Goal, TF).

partial_map(T,[],[],_,_,T) :- !.
partial_map(black('',_,_,_),Map,Map,Nil,_,Nil) :- !.
partial_map(red(L,K,V,R),Map,MapF,Nil,Goal,red(NL,K,NV,NR)) :-
	partial_map(L,Map,MapI,Nil,Goal,NL),
	(
	  MapI == [] ->
	  NR = R, NV = V, MapF = []
	;
	  MapI = [K1|MapR],
	  (
	   K == K1
	   ->
	    ( call(Goal,V,NV) -> true ; NV = V ),
	    MapN = MapR
	   ;
	    NV = V,
	    MapN = MapI
	   ),
	  partial_map(R,MapN,MapF,Nil,Goal,NR)
	).
partial_map(black(L,K,V,R),Map,MapF,Nil,Goal,black(NL,K,NV,NR)) :-
	partial_map(L,Map,MapI,Nil,Goal,NL),
	(
	  MapI == [] ->
	  NR = R, NV = V, MapF = []
	;
	  MapI = [K1|MapR],
	  (
	   K == K1
	   ->
	    ( call(Goal,V,NV) -> true ; NV = V ),
	    MapN = MapR
	   ;
	    NV = V,
	    MapN = MapI
	   ),
	  partial_map(R,MapN,MapF,Nil,Goal,NR)
	).


/** @pred rb_keys(+ _T_,+ _Keys_)


 _Keys_ is an infix visit with all keys in tree  _T_. Keys will be
sorted, but may be duplicate.


*/
rb_keys(t(_,T),Lf) :-
	keys(T,[],Lf).

rb_keys(t(_,T),L0,Lf) :-
	keys(T,L0,Lf).

keys(black('',_,_,''),L,L) :- !.
keys(red(L,K,_,R),L0,Lf) :-
	keys(L,[K|L1],Lf),
	keys(R,L0,L1).
keys(black(L,K,_,R),L0,Lf) :-
	keys(L,[K|L1],Lf),
	keys(R,L0,L1).


	%%	list_to_rbtree(+L, -T) is det.
	%
	%	T is the red-black tree corresponding to the mapping in list L.

keys_to_rbtree(List, T) :-
		sort(List,Sorted),
		ord_keys_to_rbtree(Sorted, T).

%%	list_to_rbtree(+L, -T) is det.
%
%	T is the red-black tree corresponding to the mapping in list L.

ord_keys_to_rbtree(List, T) :-
			maplist(paux, List, Sorted),
			ord_list_to_rbtree(Sorted, T).

paux(K, K-_).

			%%	list_to_rbtree(+L, -T) is det.
			%
			%	T is the red-black tree corresponding to the mapping in list L.

			list_to_rbtree(List, T) :-
				sort(List,Sorted),
				ord_list_to_rbtree(Sorted, T).

%%	ord_list_to_rbtree(+L, -T) is det.
%
%	T is the red-black tree corresponding  to the mapping in ordered
%	list L.
ord_list_to_rbtree([], t(Nil,Nil)) :- !,
	Nil = black('', _, _, '').
ord_list_to_rbtree([K-V], t(Nil,black(Nil,K,V,Nil))) :- !,
	Nil = black('', _, _, '').
ord_list_to_rbtree(List, t(Nil,Tree)) :-
	Nil = black('', _, _, ''),
	Ar =.. [seq|List],
	functor(Ar,_,L),
	Height is truncate(log(L)/log(2)),
	construct_rbtree(1, L, Ar, Height, Nil, Tree).

construct_rbtree(L, M, _, _, Nil, Nil) :- M < L, !.
construct_rbtree(L, L, Ar, Depth, Nil, Node) :- !,
	arg(L, Ar, K-Val),
	build_node(Depth, Nil, K, Val, Nil, Node).
construct_rbtree(I0, Max, Ar, Depth, Nil, Node) :-
	I is (I0+Max)//2,
	arg(I, Ar, K-Val),
	build_node(Depth, Left, K, Val, Right, Node),
	I1 is I-1,
	NewDepth is Depth-1,
	construct_rbtree(I0, I1, Ar, NewDepth, Nil, Left),
	I2 is I+1,
	construct_rbtree(I2, Max, Ar, NewDepth, Nil, Right).

build_node( 0, Left, K, Val, Right, red(Left, K, Val, Right)) :- !.
build_node( _, Left, K, Val, Right, black(Left, K, Val, Right)).


/** @pred rb_size(+ _T_,- _Size_)


 _Size_ is the number of elements in  _T_.


*/
rb_size(t(_,T),Size) :-
	size(T,0,Size).

size(black('',_,_,_),Sz,Sz) :- !.
size(red(L,_,_,R),Sz0,Szf) :-
	Sz1 is Sz0+1,
	size(L,Sz1,Sz2),
	size(R,Sz2,Szf).
size(black(L,_,_,R),Sz0,Szf) :-
	Sz1 is Sz0+1,
	size(L,Sz1,Sz2),
	size(R,Sz2,Szf).

%%	is_rbtree(?Term) is semidet.
%
%	True if Term is a valid Red-Black tree.
%
%	@tbd	Catch variables.
is_rbtree(X) :-
	var(X), !, fail.
is_rbtree(t(Nil,Nil)) :- !.
is_rbtree(t(_,T)) :-
	catch(rbtree1(T), msg(_,_), fail).

is_rbtree(X,_) :-
	var(X), !, fail.
is_rbtree(T,Goal) :-
	catch(rbtree1(T), msg(S,Args), (once(Goal),format(S,Args))).

%
% This code checks if a tree is ordered and a rbtree
%
%
rbtree(t(_,black('',_,_,''))) :- !.
rbtree(t(_,T)) :-
	catch(rbtree1(T),msg(S,Args),format(S,Args)).

rbtree1(black(L,K,_,R)) :-
	find_path_blacks(L, 0, Bls),
	check_rbtree(L,-inf,K,Bls),
	check_rbtree(R,K,+inf,Bls).
rbtree1(red(_,_,_,_)) :-
	throw(msg("root should be black",[])).


find_path_blacks(black('',_,_,''), Bls, Bls) :- !.
find_path_blacks(black(L,_,_,_), Bls0, Bls) :-
	Bls1 is Bls0+1,
	find_path_blacks(L, Bls1, Bls).
find_path_blacks(red(L,_,_,_), Bls0, Bls) :-
	find_path_blacks(L, Bls0, Bls).

check_rbtree(black('',_,_,''),Min,Max,Bls0) :- !,
	check_height(Bls0,Min,Max).
check_rbtree(red(L,K,_,R),Min,Max,Bls) :-
	check_val(K,Min,Max),
	check_red_child(L),
	check_red_child(R),
	check_rbtree(L,Min,K,Bls),
	check_rbtree(R,K,Max,Bls).
check_rbtree(black(L,K,_,R),Min,Max,Bls0) :-
	check_val(K,Min,Max),
	Bls is Bls0-1,
	check_rbtree(L,Min,K,Bls),
	check_rbtree(R,K,Max,Bls).

check_height(0,_,_) :- !.
check_height(Bls0,Min,Max) :-
	throw(msg("Unbalance ~d between ~w and ~w~n",[Bls0,Min,Max])).

check_val(K, Min, Max) :- ( K @> Min ; Min == -inf), (K @< Max ; Max == +inf), !.
check_val(K, Min, Max) :-
	throw(msg("not ordered: ~w not between ~w and ~w~n",[K,Min,Max])).

check_red_child(black(_,_,_,_)).
check_red_child(red(_,K,_,_)) :-
	throw(msg("must be red: ~w~n",[K])).


%count(1,16,X), format("deleting ~d~n",[X]), new(1,a,T0), insert(T0,2,b,T1), insert(T1,3,c,T2), insert(T2,4,c,T3), insert(T3,5,c,T4), insert(T4,6,c,T5), insert(T5,7,c,T6), insert(T6,8,c,T7), insert(T7,9,c,T8), insert(T8,10,c,T9),insert(T9,11,c,T10), insert(T10,12,c,T11),insert(T11,13,c,T12),insert(T12,14,c,T13),insert(T13,15,c,T14), insert(T14,16,c,T15),delete(T15,X,T16),pretty_print(T16),rbtree(T16),fail.

% count(1,16,X0), X is -X0, format("deleting ~d~n",[X]), new(-1,a,T0), insert(T0,-2,b,T1), insert(T1,-3,c,T2), insert(T2,-4,c,T3), insert(T3,-5,c,T4), insert(T4,-6,c,T5), insert(T5,-7,c,T6), insert(T6,-8,c,T7), insert(T7,-9,c,T8), insert(T8,-10,c,T9),insert(T9,-11,c,T10), insert(T10,-12,c,T11),insert(T11,-13,c,T12),insert(T12,-14,c,T13),insert(T13,-15,c,T14), insert(T14,-16,c,T15),delete(T15,X,T16),pretty_print(T16),rbtree(T16),fail.

count(I,_,I).
count(I,M,L) :-
	I < M, I1 is I+1, count(I1,M,L).

test_pos :-
	rb_new(1,a,T0),
	N = 10000,
	build_ptree(2,N,T0,T),
%	pretty_print(T),
	rbtree(T),
	clean_tree(1,N,T,_),
	bclean_tree(N,1,T,_),
	count(1,N,X), ( rb_delete(T,X,TF) -> true ; abort ),
%	pretty_print(TF),
	rbtree(TF),
%	format("done ~d~n",[X]),
	fail.
test_pos.

build_ptree(X,X,T0,TF) :- !,
	rb_insert(T0,X,X,TF).
build_ptree(X1,X,T0,TF) :-
	rb_insert(T0,X1,X1,TI),
	X2 is X1+1,
	build_ptree(X2,X,TI,TF).


clean_tree(X,X,T0,TF) :- !,
	rb_delete(T0,X,TF),
	( rbtree(TF) -> true ; abort).
clean_tree(X1,X,T0,TF) :-
	rb_delete(T0,X1,TI),
	X2 is X1+1,
	( rbtree(TI) -> true ; abort),
	clean_tree(X2,X,TI,TF).

bclean_tree(X,X,T0,TF) :- !,
	format("cleaning ~d~n", [X]),
	rb_delete(T0,X,TF),
	( rbtree(TF) -> true ; abort).
bclean_tree(X1,X,T0,TF) :-
	format("cleaning ~d~n", [X1]),
	rb_delete(T0,X1,TI),
	X2 is X1-1,
	( rbtree(TI) -> true ; abort),
	bclean_tree(X2,X,TI,TF).



test_neg :-
	Size = 10000,
	rb_new(-1,a,T0),
	build_ntree(2,Size,T0,T),
%	pretty_print(T),
	rbtree(T),
	MSize is -Size,
	clean_tree(MSize,-1,T,_),
	bclean_tree(-1,MSize,T,_),
	count(1,Size,X), NX is -X, ( rb_delete(T,NX,TF) -> true ; abort ),
%	pretty_print(TF),
	rbtree(TF),
%	format("done ~d~n",[X]),
	fail.
test_neg.

build_ntree(X,X,T0,TF) :- !,
	X1 is -X,
	rb_insert(T0,X1,X1,TF).
build_ntree(X1,X,T0,TF) :-
	NX1 is -X1,
	rb_insert(T0,NX1,NX1,TI),
	X2 is X1+1,
	build_ntree(X2,X,TI,TF).

/**
   @}
*/
