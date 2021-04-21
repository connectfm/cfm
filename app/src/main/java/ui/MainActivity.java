package ui;

<<<<<<< HEAD:app/src/main/java/com/example/cfm/ui/MainActivity.java
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
=======
>>>>>>> 179aec2388e0e21ea09cb9653740089b9beb0867:app/src/main/java/ui/MainActivity.java
import android.os.Build;
import android.os.Bundle;

import android.content.SharedPreferences;
<<<<<<< HEAD:app/src/main/java/com/example/cfm/ui/MainActivity.java
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
=======
>>>>>>> 179aec2388e0e21ea09cb9653740089b9beb0867:app/src/main/java/ui/MainActivity.java


import com.example.cfm.R;
import spotify_framework.Song;
import spotify_framework.SongService;


import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private SharedPreferences.Editor editor;


    private Song song;
    private SongService songService;
    private ArrayList<Song> recentlyPlayed;


    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        songService = new SongService(getApplicationContext());

        SharedPreferences preferences = this.getSharedPreferences("SPOTIFY", 0);
        BottomNavigationView navView = findViewById(R.id.nav_view);
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment);
        NavigationUI.setupWithNavController(navView, navController);

        locationTest();
    }

    private void locationTest() {

        final long LOCATION_INTERVAL = 900000;
        final long FASTEST_LOCATION_INTERVAL = LOCATION_INTERVAL;

        System.out.println("starting the test");
        LocationRequest lr = LocationRequest.create();
        lr.setInterval(LOCATION_INTERVAL);
        lr.setFastestInterval(FASTEST_LOCATION_INTERVAL);
        lr.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
        System.out.println(Build.VERSION.SDK_INT);

        final HandlerThread ht = new HandlerThread("location stuff");
        ht.start();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            System.out.println("damn pt 2");
        }

        LocationServices.getFusedLocationProviderClient(this).requestLocationUpdates(lr, new LocationCallback() {
            @Override
            public void onLocationResult(@NotNull LocationResult lr) {
                super.onLocationResult(lr);
                System.out.println("in the locationcallback");
                if (lr != null && lr.getLocations().size() > 0) {
                    int i = lr.getLocations().size();
                    System.out.println(lr.getLocations().get(i - 1));
                }
                //LocationServices.getFusedLocationProviderClient(MainActivity.this).removeLocationUpdates(this);
                //ht.quit();
            }
        }, ht.getLooper());
        System.out.println("finishing the test");
    }


}