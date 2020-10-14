using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class Movement : MonoBehaviour
{

    private NavMeshAgent agent;

    private Vector3 target;
    // Start is called before the first frame update
    void Start()
    {
        target = new Vector3(0, 0.1f, 0);
        agent = GetComponent<NavMeshAgent>();
    }

    // Update is called once per frame
    void Update()
    {
        agent.SetDestination(target);
    }
}
