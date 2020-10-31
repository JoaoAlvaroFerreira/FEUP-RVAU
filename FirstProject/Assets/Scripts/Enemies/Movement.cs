using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class Movement : MonoBehaviour
{

    private NavMeshAgent agent;

    private bool attacking;
    private Vector3 target;

    //private GameObject mainMarker;

    public Animator anim;

    // Start is called before the first frame update
    void Start()
    {
        target = new Vector3(0, 0.1f, 0);
        agent = GetComponent<NavMeshAgent>();
        anim.SetBool("isCharging", true);
        attacking = false;
    }

    // Update is called once per frame
    private void FixedUpdate() {
        if(attacking)
        Debug.Log("aaaaaaaaaa");    
    }
    void Update()
    {

        // mainMarker = GameObject.FindWithTag("MainMarker");

        // if(mainMarker.GetComponent<MainMarker>().EnteredTrigger)
        // {

        // }

        agent.SetDestination(target);

        // Check if we've reached the destination
        if (anim.GetBool("isCharging"))
        {
            if (!agent.pathPending)
            {
                if (agent.remainingDistance <= agent.stoppingDistance)
                {
                    if (!agent.hasPath || agent.velocity.sqrMagnitude == 0f)
                    {
                        Debug.Log("IS ATTACKING");
                        attack();
                    }
                }
            }
        }

    }

    void attack(){
        attacking = true;
        anim.SetBool("isCharging", false);
        anim.SetBool("isAttacking", true);
    }
}
