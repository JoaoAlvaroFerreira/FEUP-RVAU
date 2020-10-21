using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FindAndFireTower : MonoBehaviour
{
    public GameObject bullet;
    public float fire_interval = 1f;
    public float bullet_spawn_offset = 1f;
    private float time;
    // Start is called before the first frame update
    void Start()
    {
        this.gameObject.SetActive(false);
        time = 0;
    }

    // Update is called once per frame
    void Update()
    {
        time += Time.deltaTime;
        if (time >= fire_interval + 1)
            time = fire_interval;
        var enemies = GameController.getEnemies();
        float shortest_distance = float.MaxValue;
        GameObject closest_enemy = null;
        foreach(var enemy in enemies)
        {
            float distance = Vector2.Distance(new Vector2(this.transform.position.x, this.transform.position.z), new Vector2(enemy.transform.position.x, enemy.transform.position.z));
            if(shortest_distance > distance)
            {
                shortest_distance = distance;
                closest_enemy = enemy;
            }
        }
        if(closest_enemy)
        {
            var rotation = Quaternion.LookRotation(closest_enemy.transform.position - this.transform.position, Vector3.up);
            this.transform.rotation = rotation * Quaternion.Euler(0, 90, 0);
            if (time >= fire_interval)
            {
                Instantiate(bullet, this.transform.position + (rotation * Vector3.forward * bullet_spawn_offset), rotation);
                time = 0;
            }
        }
    }
}
