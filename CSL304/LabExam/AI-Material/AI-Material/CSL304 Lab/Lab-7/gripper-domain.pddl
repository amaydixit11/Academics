;; Gripper Domain Definition
(define (domain gripper-domain)
  (:requirements :strips :typing :equality)
  
  ;; Define types (if using typing)
  (:types room ball gripper)
  
  ;; Predicates
  (:predicates
    (at-robby ?r - room)           ; robot is in room ?r
    (at-ball ?b - ball ?r - room)  ; ball ?b is in room ?r
    (free ?g - gripper)             ; gripper ?g is empty
    (carry ?g - gripper ?b - ball)  ; gripper ?g holds ball ?b
  )
  
  ;; Action: Move robot between rooms
  (:action move
    :parameters (?from ?to - room)
    :precondition (and 
      (at-robby ?from)
      (not (= ?from ?to))
    )
    :effect (and 
      (at-robby ?to)
      (not (at-robby ?from))
    )
  )
  
  ;; Action: Pick up a ball
  (:action pick-up
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and 
      (at-ball ?b ?r)
      (at-robby ?r)
      (free ?g)
    )
    :effect (and 
      (carry ?g ?b)
      (not (at-ball ?b ?r))
      (not (free ?g))
    )
  )
  
  ;; Action: Drop a ball
  (:action drop
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and 
      (carry ?g ?b)
      (at-robby ?r)
    )
    :effect (and 
      (at-ball ?b ?r)
      (free ?g)
      (not (carry ?g ?b))
    )
  )
)