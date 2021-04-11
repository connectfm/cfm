package com.example.cfm;

import android.Manifest;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Criteria;
import android.location.Location;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;

import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.location.Location;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.android.volley.RequestQueue;
import com.android.volley.toolbox.Volley;
import com.example.spotify_framework.Song;
import com.example.spotify_framework.SongService;

//import com.android.volley.RequestQueue;
//import com.android.volley.toolbox.Volley;

//import com.android.volley.RequestQueue;
//import com.android.volley.toolbox.Volley;
import com.example.spotify_framework.UserService;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.spotify.sdk.android.auth.AuthorizationClient;
import com.spotify.sdk.android.auth.AuthorizationRequest;
import com.spotify.sdk.android.auth.AuthorizationResponse;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private SharedPreferences.Editor editor;

    private TextView userView;
    private TextView songView;
    private Button addButton;
    private Button logoutButton;
    private ImageView albums;

    private Song song;
    private SongService songService;
    private ArrayList<Song> recentlyPlayed;


    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        System.out.println("debugging stuff");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        songService = new SongService(getApplicationContext());
        userView = (TextView) findViewById(R.id.user);
        songView = (TextView) findViewById(R.id.song);
        addButton = (Button) findViewById(R.id.add);
        logoutButton = (Button) findViewById(R.id.logout);

        SharedPreferences preferences = this.getSharedPreferences("SPOTIFY", 0);
        userView.setText("Welcome to connect.fm "+ preferences.getString("displayName", "NoName"));
        BottomNavigationView navView = findViewById(R.id.nav_view);
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment);
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(navView, navController);

        getTracks();


        logoutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                logOut();
            }
        });
    }


    private void getTracks() {
        Log.d("STARTING: ", "Getting Tracks");
        songService.getRecentlyPlayed(() -> {
            recentlyPlayed = songService.getPlaylist();
            updateSong();
        });
    }

    private void updateSong() {
        if(recentlyPlayed.size() > 0) {
            songView.setText(recentlyPlayed.get(0).getTitle());
            song = recentlyPlayed.get(0);
        }
    }

    private void logOut() {
        editor = getSharedPreferences("SPOTIFY",0).edit();
        editor.clear();
        Intent intent = new Intent(MainActivity.this, SplashActivity.class);
        startActivity(intent);
    }

}