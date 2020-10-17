using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionWithBullet : MonoBehaviour
{
    public GameObject enemyInstance;
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnTriggerEnter(Collider other) 
    {
        Debug.Log(other.gameObject.tag);
        Debug.Log("aaa");
        if (other.gameObject.tag == "Bullet")
        {
            Destroy(other.gameObject);
            Destroy(this.enemyInstance);
            //or gameObject.SetActive(false);
        }
    }
}
