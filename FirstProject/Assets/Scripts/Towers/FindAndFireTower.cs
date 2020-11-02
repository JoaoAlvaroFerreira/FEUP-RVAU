using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class FindAndFireTower : MonoBehaviour
{
    public GameObject bullet;
    public float fire_interval = 0.5f;

    public int bulletsLeft = 20;
    public Slider bulletUi;

    public void replenishBullets()
    {
        bulletsLeft +=5;
    }

    public float bullet_spawn_offset = 1f;
    private float time;

    private Transform head;
    private Transform turret;
    // Start is called before the first frame update
    void Start()
    {
        this.gameObject.SetActive(false);
        time = 0;
        this.head = this.transform.Find("Base/Pylon");
        this.turret = this.transform.Find("Base/Pylon/Turret");
    }

    // Update is called once per frame
    void Update()
    {
        bulletUi.value = bulletsLeft;
        
        time += Time.deltaTime;
        if (time >= fire_interval + 1)
            time = fire_interval;
        var enemies = WaveSpawner.getEnemiesFromWave();
        float shortest_distance = float.MaxValue;
        GameObject closest_enemy = null;

        foreach (var enemy in enemies)
        {
            float distance = Vector2.Distance(new Vector2(this.turret.transform.position.x, this.turret.transform.position.z), new Vector2(enemy.transform.position.x, enemy.transform.position.z));
            if (shortest_distance > distance)
            {
                shortest_distance = distance;
                closest_enemy = enemy;
            }
        }
        if (closest_enemy)
        {
            var rotation = Quaternion.LookRotation(closest_enemy.transform.position - this.turret.transform.position, Vector3.up);
            this.turret.rotation = rotation * Quaternion.Euler(0, -90, 0);
            if (time >= fire_interval && bulletsLeft > 0)
            {
                Instantiate(bullet, this.turret.transform.position + (rotation * Vector3.forward * bullet_spawn_offset), rotation);
                time = 0;
                bulletsLeft -= 1;
            }
        }
    }
}
