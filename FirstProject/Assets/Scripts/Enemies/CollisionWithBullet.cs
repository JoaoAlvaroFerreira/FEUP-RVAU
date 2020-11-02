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
        if (other.gameObject.tag == "Bullet")
        {
            Destroy(other.gameObject);
            WaveSpawner.deleteEnemyFromWave(gameObject);
            Destroy(gameObject);
            //or gameObject.SetActive(false);
        }
    }
}
