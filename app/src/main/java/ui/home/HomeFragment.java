package ui.home;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.cfm.recommend.Recommender;
import com.example.cfm.R;

import spotify_framework.Song;
import spotify_framework.SongService;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static java.lang.Thread.sleep;

public class HomeFragment extends Fragment {
    public SongService songService;
    private RecyclerView recyclerView;
    private SongsAdapter adapter;
    private Recommender recommender;
    private Button radioStart;

    public void onCreate() {
    }

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_home, container, false);
        radioStart = (Button) root.findViewById(R.id.begin_listening);
        radioStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            }
        });
        return root;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        songService = new SongService(getActivity());

        songService.getRecentlyPlayed(() -> {
            ArrayList<Song> recents = songService.getPlaylist();
            adapter = new SongsAdapter(getActivity(), recents);
            adapter.notifyDataSetChanged();
            recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
            recyclerView.setAdapter(adapter);
            recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
        });
    }

}