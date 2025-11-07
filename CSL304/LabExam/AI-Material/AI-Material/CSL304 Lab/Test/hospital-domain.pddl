;; Hospital Emergency Domain Definition
(define (domain hospital-emergency)
  (:requirements :strips :typing :negative-preconditions)
  
  ;; Define types
  (:types 
    ambulance patient location - object
    hospital - location
  )
  
  ;; Predicates
  (:predicates
    (at ?a - ambulance ?l - location)        ; ambulance ?a is at location ?l
    (patient-at ?p - patient ?l - location)  ; patient ?p is at location ?l
    (in-ambulance ?p - patient ?a - ambulance) ; patient ?p is inside ambulance ?a
    (connected ?l1 ?l2 - location)           ; locations are connected
    (blocked ?l1 ?l2 - location)            ; route is blocked
    (empty ?a - ambulance)                   ; ambulance is empty
  )
  
  ;; Action: Move ambulance between connected locations
  (:action move
    :parameters (?a - ambulance ?from ?to - location)
    :precondition (and 
      (at ?a ?from)
      (connected ?from ?to)
      (not (blocked ?from ?to))
    )
    :effect (and 
      (at ?a ?to)
      (not (at ?a ?from))
    )
  )
  
  ;; Action: Pick up a patient
  (:action pickup-patient
    :parameters (?p - patient ?a - ambulance ?l - location)
    :precondition (and 
      (at ?a ?l)
      (patient-at ?p ?l)
      (empty ?a)
    )
    :effect (and 
      (in-ambulance ?p ?a)
      (not (patient-at ?p ?l))
      (not (empty ?a))
    )
  )
  
  ;; Action: Drop off a patient at hospital
  (:action dropoff-patient
    :parameters (?p - patient ?a - ambulance ?h - hospital)
    :precondition (and 
      (at ?a ?h)
      (in-ambulance ?p ?a)
    )
    :effect (and 
      (patient-at ?p ?h)
      (empty ?a)
      (not (in-ambulance ?p ?a))
    )
  )
)